package ai.koog.prompt.executor.clients.lmstudio

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.agents.utils.SuitableForIO
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.executor.clients.LLMEmbeddingProvider
import ai.koog.prompt.executor.clients.lmstudio.LMStudioToolChoice.FunctionName
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.MediaContent
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.prompt.params.LLMParams
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.plugins.sse.SSE
import io.ktor.client.plugins.sse.SSEClientException
import io.ktor.client.plugins.sse.sse
import io.ktor.client.request.accept
import io.ktor.client.request.header
import io.ktor.client.request.headers
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.client.statement.readRawBytes
import ai.koog.prompt.executor.clients.lmstudio.LMStudioAPIException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioAuthenticationException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioClientErrorException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioErrorDetails
import ai.koog.prompt.executor.clients.lmstudio.LMStudioMalformedResponseException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioNotFoundException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioRateLimitException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioServerException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioStreamException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioConnectionException
import ai.koog.prompt.executor.clients.lmstudio.LMStudioStreamEvent // Added import
import io.ktor.client.plugins.ResponseException
import kotlinx.serialization.SerializationException
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.mapNotNull
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock
import kotlinx.serialization.json.ClassDiscriminatorMode
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNamingStrategy
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlin.io.encoding.Base64
import kotlin.io.encoding.ExperimentalEncodingApi
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Represents the settings for configuring an LMStudio client.
 *
 * @property baseUrl The base URL of the LMStudio API. Defaults to "http://localhost:1234".
 * @property timeoutConfig Configuration for connection timeouts, including request, connect, and socket timeouts.
 * @property chatCompletionsPath The path of the LMStudio Chat Completions API. Defaults to "api/v0/chat/completions".
 * @property embeddingsPath The path of the LMStudio Embeddings API. Defaults to "api/v0/embeddings".
 * @property modelsPath The path of the LMStudio Models API. Defaults to "api/v0/models".
 */
public class LMStudioClientSettings(
    public val baseUrl: String = "http://localhost:1234",
    public val timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig(),
    public val chatCompletionsPath: String = "api/v0/chat/completions",
    public val embeddingsPath: String = "api/v0/embeddings",
    public val modelsPath: String = "api/v0/models",
)

/**
 * Implementation of [LLMClient] for LMStudio API.
 * Uses Ktor HttpClient to communicate with the LMStudio API.
 *
 * @param settings The base URL and timeouts for the LMStudio API, defaults to "http://localhost:1234" and 900 s
 * @param clock Clock instance used for tracking response metadata timestamps.
 */
public open class LMStudioLLMClient(
    private val settings: LMStudioClientSettings = LMStudioClientSettings(),
    baseClient: HttpClient = HttpClient(),
    private val clock: Clock = Clock.System,
) : LLMEmbeddingProvider, LLMClient {

    private companion object {
        private val logger = KotlinLogging.logger { }
    }

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        encodeDefaults = true
        explicitNulls = false
        namingStrategy = JsonNamingStrategy.SnakeCase
        // LMStudio API is not polymorphic, it's "dynamic". Don't add polymorphic discriminators
        classDiscriminatorMode = ClassDiscriminatorMode.NONE
    }

    private val httpClient = baseClient.config {
        defaultRequest {
            url(settings.baseUrl)
            contentType(ContentType.Application.Json)
        }
        install(SSE)
        install(ContentNegotiation) {
            json(json)
        }
        install(HttpTimeout) {
            requestTimeoutMillis = settings.timeoutConfig.requestTimeoutMillis // Increase timeout to 60 seconds
            connectTimeoutMillis = settings.timeoutConfig.connectTimeoutMillis
            socketTimeoutMillis = settings.timeoutConfig.socketTimeoutMillis
        }
    }

    override suspend fun execute(prompt: Prompt, model: LLModel, tools: List<ToolDescriptor>): List<Message.Response> {
        logger.debug { "Executing prompt: $prompt with tools: $tools and model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        require(model.capabilities.contains(LLMCapability.Tools) || tools.isEmpty()) {
            "Model ${model.id} does not support tools"
        }

        val request = createLMStudioRequest(prompt, tools, model, false)

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(settings.chatCompletionsPath) {
                setBody(request)
            }

            if (response.status.isSuccess()) {
                try {
                    val openAIResponse = response.body<LMStudioResponse>()
                    processLMStudioResponse(openAIResponse)
                } catch (e: SerializationException) {
                    throw LMStudioMalformedResponseException("Failed to parse successful response: ${e.message}", e)
                }
            } else {
                val errorBody = response.bodyAsText()
                val httpStatusCode = response.status.value
                var errorDetails: LMStudioErrorDetails? = null
                try {
                    if (errorBody.isNotBlank()) {
                        errorDetails = json.decodeFromString<LMStudioErrorDetails>(errorBody)
                    }
                } catch (e: SerializationException) {
                    logger.warn { "Failed to parse LMStudio JSON error response: $errorBody" }
                }

                val errorMessage = errorDetails?.getDetailedMessage() ?: errorBody
                val logMessage = "Error from LMStudio API: $httpStatusCode - $errorMessage. Details: $errorDetails"
                logger.error { logMessage }

                when (httpStatusCode) {
                    401, 403 -> throw LMStudioAuthenticationException(errorJson = errorBody, message = errorMessage)
                    404 -> throw LMStudioNotFoundException(errorJson = errorBody, message = errorMessage)
                    429 -> {
                        // Placeholder for Retry-After header extraction if needed
                        val retryAfter = response.headers[HttpHeaders.RetryAfter]?.toIntOrNull()
                        throw LMStudioRateLimitException(errorJson = errorBody, message = errorMessage, retryAfterSeconds = retryAfter)
                    }
                    in 400..499 -> throw LMStudioClientErrorException(httpStatusCode, errorJson = errorBody, message = errorMessage)
                    in 500..599 -> throw LMStudioServerException(httpStatusCode, errorJson = errorBody, message = errorMessage)
                    else -> throw LMStudioAPIException(httpStatusCode, errorJson = errorBody, message = "Unhandled API error: $errorMessage")
                }
            }
        }
    }

    override fun executeStreaming(prompt: Prompt, model: LLModel): Flow<LMStudioStreamEvent> = flow {
        logger.debug { "Executing streaming prompt: $prompt with model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }

        val request = createLMStudioRequest(prompt, emptyList(), model, true)

        httpClient.sse(
            urlString = settings.chatCompletionsPath,
            request = {
                method = HttpMethod.Post
                accept(ContentType.Text.EventStream)
                headers {
                    append(HttpHeaders.CacheControl, "no-cache")
                    append(HttpHeaders.Connection, "keep-alive")
                }
                setBody(request)
            }
        ) { // SSEClientSession
            incoming.collect { event ->
                event.data?.trim()?.takeIf { it.isNotEmpty() }?.let { dataString ->
                    if (dataString == "[DONE]") {
                        // Stream is done, collector will complete.
                        return@let
                    }
                    try {
                        val streamResponse = json.decodeFromString<LMStudioStreamResponse>(dataString)
                        streamResponse.choices.forEach { choice ->
                            // Emit text content if present
                            choice.delta.content?.let { contentChunk ->
                                if (contentChunk.isNotEmpty()) { // Avoid emitting empty strings if not desired
                                     emit(LMStudioStreamEvent.TextChunk(contentChunk))
                                }
                            }
                            // Emit tool call chunks if present
                            choice.delta.toolCalls?.forEachIndexed { index, toolCall -> // toolCall is LMStudioToolCall
                                emit(LMStudioStreamEvent.ToolCallChunk(
                                    toolCallId = toolCall.id,
                                    functionName = toolCall.function.name,
                                    argumentsChunk = toolCall.function.arguments,
                                    toolCallIndex = index
                                ))
                            }
                            // Emit finish reason if present in this delta
                            choice.finishReason?.let { reason ->
                                emit(LMStudioStreamEvent.FinishReason(reason))
                            }
                        }
                    } catch (e: SerializationException) {
                        logger.warn(e) { "Error parsing LMStudio stream event data: $dataString" }
                        // This exception will be caught by the .catch operator on the flow
                        throw LMStudioMalformedResponseException("Error parsing stream event: ${e.message}", e)
                    }
                }
            }
        }
    }.catch { cause ->
        when (cause) {
            is LMStudioMalformedResponseException -> throw cause // Re-throw if already specific
            is SerializationException -> { // Catch parsing errors not caught inside collect (less likely here)
                logger.error(cause) { "Serialization error during streaming setup or event processing: ${cause.message}" }
                throw LMStudioMalformedResponseException("Serialization error in stream: ${cause.message}", cause)
            }
            is SSEClientException -> {
                val response = cause.response
                val httpStatusCode = response?.status?.value
                var errorBody: String? = null
                var errorDetails: LMStudioErrorDetails? = null
                var errorMessage = cause.message ?: "SSE connection error"

                if (response != null) {
                    try {
                        errorBody = response.bodyAsText() // Ktor 2.x, Ktor 1.x might need readRawBytes
                        if (errorBody.isNotBlank()) {
                           errorDetails = json.decodeFromString<LMStudioErrorDetails>(errorBody)
                           errorMessage = errorDetails.getDetailedMessage() ?: errorMessage
                        }
                    } catch (se: SerializationException) {
                        logger.warn(se) { "Failed to parse LMStudio JSON error from SSE response: $errorBody" }
                    } catch (ioe: Exception) {
                        logger.warn(ioe) { "Failed to read or parse error body from SSE response: ${ioe.message}" }
                    }
                }
                val logMessage = "SSE Error from LMStudio API: $httpStatusCode - $errorMessage. Details: $errorDetails"
                logger.error(cause) { logMessage }

                if (httpStatusCode != null) {
                    when (httpStatusCode) {
                        401, 403 -> throw LMStudioAuthenticationException(errorJson = errorBody, message = errorMessage, cause = cause)
                        404 -> throw LMStudioNotFoundException(errorJson = errorBody, message = errorMessage, cause = cause)
                        429 -> {
                            val retryAfter = response?.headers?.get(HttpHeaders.RetryAfter)?.toIntOrNull()
                            throw LMStudioRateLimitException(errorJson = errorBody, message = errorMessage, retryAfterSeconds = retryAfter, cause = cause)
                        }
                        in 400..499 -> throw LMStudioClientErrorException(httpStatusCode, errorJson = errorBody, message = errorMessage, cause = cause)
                        in 500..599 -> throw LMStudioServerException(httpStatusCode, errorJson = errorBody, message = errorMessage, cause = cause)
                        else -> throw LMStudioStreamException("SSE error with code $httpStatusCode: $errorMessage", cause = cause)
                    }
                } else {
                    throw LMStudioStreamException("SSE error: $errorMessage", cause = cause)
                }
            }
            is ResponseException -> { // Catches Ktor client errors for initial HTTP request (non-2xx)
                val httpStatusCode = cause.response.status.value
                val errorBody = cause.response.bodyAsText()
                var errorDetails: LMStudioErrorDetails? = null
                try {
                    if (errorBody.isNotBlank()) {
                        errorDetails = json.decodeFromString<LMStudioErrorDetails>(errorBody)
                    }
                } catch (e: SerializationException) {
                    logger.warn { "Failed to parse LMStudio JSON error response from ResponseException: $errorBody" }
                }
                val errorMessage = errorDetails?.getDetailedMessage() ?: errorBody
                val logMessage = "Error from LMStudio API (initial stream request): $httpStatusCode - $errorMessage. Details: $errorDetails"
                logger.error(cause) { logMessage }

                when (httpStatusCode) {
                    401, 403 -> throw LMStudioAuthenticationException(errorJson = errorBody, message = errorMessage, cause = cause)
                    404 -> throw LMStudioNotFoundException(errorJson = errorBody, message = errorMessage, cause = cause)
                    429 -> {
                         val retryAfter = cause.response.headers[HttpHeaders.RetryAfter]?.toIntOrNull()
                         throw LMStudioRateLimitException(errorJson = errorBody, message = errorMessage, retryAfterSeconds = retryAfter, cause = cause)
                    }
                    in 400..499 -> throw LMStudioClientErrorException(httpStatusCode, errorJson = errorBody, message = errorMessage, cause = cause)
                    in 500..599 -> throw LMStudioServerException(httpStatusCode, errorJson = errorBody, message = errorMessage, cause = cause)
                    else -> throw LMStudioAPIException(httpStatusCode, errorJson = errorBody, message = "Unhandled API error (initial stream request): $errorMessage", cause = cause)
                }
            }
            is Exception -> { // General catch-all
                logger.error(cause) { "Unhandled exception during streaming: ${cause.message}" }
                throw LMStudioConnectionException("Unhandled exception in stream: ${cause.message}", cause)
            }
        }
    }

    /**
     * Embeds the given text using the LMStudio embeddings API.
     *
     * @param text The text to embed.
     * @param model The model to use for embedding. Must have the Embed capability.
     * @return A list of floating-point values representing the embedding.
     * @throws IllegalArgumentException if the model does not have the Embed capability.
     */
    override suspend fun embed(text: String, model: LLModel): List<Double> {
        require(model.capabilities.contains(LLMCapability.Embed)) {
            "Model ${model.id} does not have the Embed capability"
        }
        logger.debug { "Embedding text with model: ${model.id}" }

        val request = LMStudioEmbeddingRequest(
            model = model.id,
            input = text
        )

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(settings.embeddingsPath) {
                setBody(request)
            }

            if (response.status.isSuccess()) {
                try {
                    val openAIResponse = response.body<LMStudioEmbeddingResponse>()
                    if (openAIResponse.data.isNotEmpty()) {
                        openAIResponse.data.first().embedding
                    } else {
                        logger.error { "Empty data in LMStudio embedding response" }
                        throw LMStudioMalformedResponseException("Empty data in LMStudio embedding response")
                    }
                } catch (e: SerializationException) {
                    throw LMStudioMalformedResponseException("Failed to parse successful embedding response: ${e.message}", e)
                }
            } else {
                val errorBody = response.bodyAsText()
                val httpStatusCode = response.status.value
                var errorDetails: LMStudioErrorDetails? = null
                try {
                    if (errorBody.isNotBlank()) {
                        errorDetails = json.decodeFromString<LMStudioErrorDetails>(errorBody)
                    }
                } catch (e: SerializationException) {
                    logger.warn { "Failed to parse LMStudio JSON error response for embeddings: $errorBody" }
                }

                val errorMessage = errorDetails?.getDetailedMessage() ?: errorBody
                val logMessage = "Error from LMStudio Embeddings API: $httpStatusCode - $errorMessage. Details: $errorDetails"
                logger.error { logMessage }

                when (httpStatusCode) {
                    401, 403 -> throw LMStudioAuthenticationException(errorJson = errorBody, message = errorMessage)
                    404 -> throw LMStudioNotFoundException(errorJson = errorBody, message = errorMessage)
                    429 -> {
                        val retryAfter = response.headers[HttpHeaders.RetryAfter]?.toIntOrNull()
                        throw LMStudioRateLimitException(errorJson = errorBody, message = errorMessage, retryAfterSeconds = retryAfter)
                    }
                    in 400..499 -> throw LMStudioClientErrorException(httpStatusCode, errorJson = errorBody, message = errorMessage)
                    in 500..599 -> throw LMStudioServerException(httpStatusCode, errorJson = errorBody, message = errorMessage)
                    else -> throw LMStudioAPIException(httpStatusCode, errorJson = errorBody, message = "Unhandled Embeddings API error: $errorMessage")
                }
            }
        }
    }

    /**
     * Returns the list of models available in the LM Studio server.
     */
    public suspend fun getModels(): List<LMStudioModelInfo> = withContext(Dispatchers.SuitableForIO) {
        httpClient.get(settings.modelsPath).body<LMStudioModelsListResponse>().data
    }

    @OptIn(ExperimentalUuidApi::class)
    private fun createLMStudioRequest(
        prompt: Prompt,
        tools: List<ToolDescriptor>,
        model: LLModel,
        stream: Boolean
    ): LMStudioRequest {
        val messages = mutableListOf<LMStudioMessage>()
        val pendingCalls = mutableListOf<LMStudioToolCall>()

        fun flushCalls() {
            if (pendingCalls.isNotEmpty()) {
                messages += LMStudioMessage(role = "assistant", toolCalls = pendingCalls.toList())
                pendingCalls.clear()
            }
        }

        for (message in prompt.messages) {
            when (message) {
                is Message.System -> {
                    flushCalls()
                    messages.add(
                        LMStudioMessage(
                            role = "system",
                            content = Content.Text(message.content)
                        )
                    )
                }

                is Message.User -> {
                    flushCalls()
                    messages.add(message.toLMStudioMessage(model))
                }

                is Message.Assistant -> {
                    flushCalls()
                    messages.add(
                        LMStudioMessage(
                            role = "assistant",
                            content = Content.Text(message.content)
                        )
                    )
                }

                is Message.Tool.Result -> {
                    flushCalls()
                    messages.add(
                        LMStudioMessage(
                            role = "tool",
                            content = Content.Text(message.content),
                            toolCallId = message.id
                        )
                    )
                }

                is Message.Tool.Call -> pendingCalls += LMStudioToolCall(
                    id = message.id ?: Uuid.random().toString(),
                    function = LMStudioFunction(message.tool, message.content)
                )
            }
        }
        flushCalls()

        val lmStudioTools = tools.map { tool ->
            val propertiesMap = mutableMapOf<String, JsonElement>()

            // Add required parameters
            tool.requiredParameters.forEach { param ->
                propertiesMap[param.name] = buildLMStudioParam(param)
            }

            // Add optional parameters
            tool.optionalParameters.forEach { param ->
                propertiesMap[param.name] = buildLMStudioParam(param)
            }

            val parametersObject = buildJsonObject {
                put("type", JsonPrimitive("object"))
                put("properties", JsonObject(propertiesMap))
                put("required", buildJsonArray {
                    tool.requiredParameters.forEach { param ->
                        add(JsonPrimitive(param.name))
                    }
                })
            }

            LMStudioTool(
                function = LMStudioToolFunction(
                    name = tool.name,
                    description = tool.description,
                    parameters = parametersObject
                )
            )
        }

        val toolChoice = when (val toolChoice = prompt.params.toolChoice) {
            LLMParams.ToolChoice.Auto -> LMStudioToolChoice.Auto
            LLMParams.ToolChoice.None -> LMStudioToolChoice.None
            LLMParams.ToolChoice.Required -> LMStudioToolChoice.Required
            is LLMParams.ToolChoice.Named -> LMStudioToolChoice.Function(function = FunctionName(toolChoice.name))
            null -> null
        }

        val modalities = if (model.capabilities.contains(LLMCapability.Audio)) listOf(
            LMStudioModalities.Text,
            LMStudioModalities.Audio
        ) else null
        // TODO allow passing this externally and actually controlling this behavior
        val audio = modalities?.let {
            LMStudioAudioConfig(
                format = if (stream) LMStudioAudioFormat.PCM16 else LMStudioAudioFormat.WAV,
                voice = LMStudioAudioVoice.Alloy,
            )
        }

        return LMStudioRequest(
            model = model.id,
            messages = messages,
            temperature = if (model.capabilities.contains(LLMCapability.Temperature)) prompt.params.temperature else null,
            tools = if (tools.isNotEmpty()) lmStudioTools else null,
            modalities = modalities,
            audio = audio,
            stream = stream,
            toolChoice = toolChoice,
        )
    }

    private fun Message.User.toLMStudioMessage(model: LLModel): LMStudioMessage {
        val listOfContent = buildList {
            if (content.isNotEmpty() || mediaContent.isEmpty()) {
                add(ContentPart.Text(content))
            }

            mediaContent.forEach { media ->
                when (media) {
                    is MediaContent.Image -> {
                        require(model.capabilities.contains(LLMCapability.Vision.Image)) {
                            "Model ${model.id} does not support image"
                        }
                        val imageUrl = if (media.isUrl()) {
                            media.source
                        } else {
                            require(media.format in listOf("png", "jpg", "jpeg", "webp", "gif")) {
                                "Image format ${media.format} not supported"
                            }
                            "data:${media.getMimeType()};base64,${media.toBase64()}"
                        }
                        add(ContentPart.Image(ContentPart.ImageUrl(imageUrl)))
                    }

                    is MediaContent.Audio -> {
                        require(model.capabilities.contains(LLMCapability.Audio)) {
                            "Model ${model.id} does not support audio"
                        }

                        require(media.format in listOf("wav", "mp3")) {
                            "Audio format ${media.format} not supported"
                        }
                        add(ContentPart.Audio(ContentPart.InputAudio(media.toBase64(), media.format)))
                    }

                    is MediaContent.File -> {
                        require(model.capabilities.contains(LLMCapability.Vision.Document)) {
                            "Model ${model.id} does not support document files"
                        }

                        require(media.format in listOf("pdf", "txt", "md")) {
                            "File format ${media.format} not supported. Supported formats: `pdf`, `txt`, `md`"
                        }
                        val fileData = "data:${media.getMimeType()};base64,${media.toBase64()}"
                        add(
                            ContentPart.File(
                                ContentPart.FileData(
                                    fileData = fileData,
                                    filename = media.fileName()
                                )
                            )
                        )
                    }

                    else -> throw IllegalArgumentException("Unsupported media content: $media")
                }
            }
        }

        return LMStudioMessage(role = "user", content = Content.Parts(listOfContent))
    }

    private fun buildLMStudioParam(param: ToolParameterDescriptor): JsonObject = buildJsonObject {
        put("description", JsonPrimitive(param.description))
        fillLMStudioParamType(param.type)
    }

    private fun JsonObjectBuilder.fillLMStudioParamType(type: ToolParameterType) {
        when (type) {
            ToolParameterType.Boolean -> put("type", JsonPrimitive("boolean"))
            ToolParameterType.Float -> put("type", JsonPrimitive("number"))
            ToolParameterType.Integer -> put("type", JsonPrimitive("integer"))
            ToolParameterType.String -> put("type", JsonPrimitive("string"))
            is ToolParameterType.Enum -> {
                put("type", JsonPrimitive("string"))
                put("enum", buildJsonArray {
                    type.entries.forEach { entry ->
                        add(JsonPrimitive(entry))
                    }
                })
            }

            is ToolParameterType.List -> {
                put("type", JsonPrimitive("array"))
                put("items", buildJsonObject {
                    fillLMStudioParamType(type.itemsType)
                })
            }

            is ToolParameterType.Object -> {
                put("type", JsonPrimitive("object"))
                type.additionalProperties?.let {
                    put("additionalProperties", type.additionalProperties)
                }
                put("properties", buildJsonObject {
                    type.properties.forEach { property ->
                        put(property.name, buildJsonObject {
                            fillLMStudioParamType(property.type)
                            put("description", property.description)
                        })
                    }
                })
            }
        }
    }

    @OptIn(ExperimentalEncodingApi::class)
    private fun processLMStudioResponse(response: LMStudioResponse): List<Message.Response> {
        if (response.choices.isEmpty()) {
            // logger.error { "Empty choices in LMStudio response" } // This will be an exception from the caller
            throw LMStudioMalformedResponseException("Empty choices in LMStudio response")
        }

        val (choice, message) = response.choices
            .firstOrNull()
            ?.let { it to it.message } ?: throw IllegalStateException("No choice found in LMStudio response")

        // Extract token count from the response
        val totalTokensCount = response.usage?.totalTokens
        val inputTokensCount = response.usage?.inputTokens
        val outputTokensCount = response.usage?.outputTokens

        return when {
            message.toolCalls != null && message.toolCalls.isNotEmpty() -> {
                message.toolCalls.map { toolCall ->
                    Message.Tool.Call(
                        id = toolCall.id,
                        tool = toolCall.function.name,
                        content = toolCall.function.arguments,
                        metaInfo = ResponseMetaInfo.create(
                            clock,
                            totalTokensCount = totalTokensCount,
                            inputTokensCount = inputTokensCount,
                            outputTokensCount = outputTokensCount
                        )
                    )
                }
            }

            message.content != null -> {
                listOf(
                    Message.Assistant(
                        content = message.content.text(),
                        finishReason = choice.finishReason,
                        metaInfo = ResponseMetaInfo.create(
                            clock, totalTokensCount = totalTokensCount,
                            inputTokensCount = inputTokensCount,
                            outputTokensCount = outputTokensCount
                        )
                    )
                )
            }

            message.audio != null -> {
                val audioData = Base64.decode(message.audio.data)
                listOf(
                    Message.Assistant(
                        content = message.audio.transcript ?: "",
                        mediaContent = MediaContent.Audio(audioData, format = message.audio.format?.takeIf { it.isNotBlank() } ?: "wav"),
                        finishReason = choice.finishReason,
                        metaInfo = ResponseMetaInfo.create(
                            clock, totalTokensCount = totalTokensCount,
                            inputTokensCount = inputTokensCount,
                            outputTokensCount = outputTokensCount
                        )
                    )
                )
            }

            else -> {
                logger.error { "Unexpected response from LMStudio: no tool calls and no content" }
                error("Unexpected response from LMStudio: no tool calls and no content")
            }
        }
    }
}
