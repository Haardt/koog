package ai.koog.prompt.executor.clients.lmstudio

import kotlinx.serialization.*
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.descriptors.PolymorphicKind
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.*
import kotlin.jvm.JvmInline

@Serializable
internal data class LMStudioRequest(
    val model: String,
    val messages: List<LMStudioMessage>,
    val temperature: Double? = null,
    val tools: List<LMStudioTool>? = null,
    val modalities: List<LMStudioModalities>? = null,
    val audio: LMStudioAudioConfig? = null,
    val stream: Boolean = false,
    val toolChoice: LMStudioToolChoice? = null
)

@Serializable
internal enum class LMStudioModalities {
    @SerialName("text")
    Text,
    @SerialName("audio")
    Audio,
}

@Serializable
internal data class LMStudioMessage(
    val role: String,
    @Serializable(with = ContentSerializer::class)
    val content: Content? = null,
    val audio: LMStudioAudio? = null,
    val toolCalls: List<LMStudioToolCall>? = null,
    val name: String? = null,
    val toolCallId: String? = null
)

@Serializable
internal sealed interface Content {
    fun text(): String

    @JvmInline
    value class Text(val value: String) : Content {
        override fun text(): String = value
    }

    @JvmInline
    value class Parts(val value: List<ContentPart>) : Content {
        override fun text(): String = value
            .filterIsInstance<ContentPart.Text>()
            .joinToString("\n") { it.text }
    }
}

@Serializable
internal sealed interface ContentPart {
    val type: String

    @Serializable
    data class Text(val text: String, override val type: String = "text") : ContentPart

    @Serializable
    data class Image(val imageUrl: ImageUrl, override val type: String = "image_url") : ContentPart

    @Serializable
    data class ImageUrl(val url: String)

    @Serializable
    data class Audio(val inputAudio: InputAudio, override val type: String = "input_audio") : ContentPart

    /**
     * @property data Base64 encoded audio data
     * @property format The format of the encoded audio data. Currently, it supports "wav" and "mp3"
     */
    @Serializable
    data class InputAudio(val data: String, val format: String)

    @Serializable
    data class File(val file: FileData, override val type: String = "file") : ContentPart

    @Serializable
    data class FileData(val fileData: String?, val fileId: String? = null, val filename: String? = null)
}


@Serializable
internal data class LMStudioToolCall(
    val id: String,
    val type: String = "function",
    val function: LMStudioFunction
)

@Serializable
internal data class LMStudioFunction(
    val name: String,
    val arguments: String
)

@Serializable
internal data class LMStudioTool(
    val type: String = "function",
    val function: LMStudioToolFunction
)

@Serializable
internal data class LMStudioToolFunction(
    val name: String,
    val description: String,
    val parameters: JsonObject
)

@Serializable
internal data class LMStudioResponse(
    val id: String,
    @SerialName("object") val objectType: String,
    val created: Long,
    val model: String,
    val choices: List<LMStudioChoice>,
    val usage: LMStudioUsage? = null
)

@Serializable
internal data class LMStudioChoice(
    val index: Int,
    val message: LMStudioMessage,
    val finishReason: String? = null
)

@Serializable
internal data class LMStudioUsage(
    val inputTokens: Int? = null,
    val outputTokens: Int? = null,
    val totalTokens: Int
)

@Serializable
internal data class LMStudioEmbeddingRequest(
    val model: String,
    val input: String
)

@Serializable
internal data class LMStudioEmbeddingResponse(
    val data: List<LMStudioEmbeddingData>,
    val model: String,
    val usage: LMStudioUsage? = null
)

@Serializable
internal data class LMStudioEmbeddingData(
    val embedding: List<Double>,
    val index: Int
)

@Serializable
internal data class LMStudioStreamResponse(
    val id: String,
    @SerialName("object") val objectType: String,
    val created: Long,
    val model: String,
    val choices: List<LMStudioStreamChoice>
)

@Serializable
internal data class LMStudioStreamChoice(
    val index: Int,
    val delta: LMStudioStreamDelta,
    val finishReason: String? = null
)

@Serializable
internal data class LMStudioStreamDelta(
    val role: String? = null,
    val content: String? = null,
    val toolCalls: List<LMStudioToolCall>? = null
)

@Serializable
internal sealed interface LMStudioToolChoice {
    @JvmInline
    @Serializable
    value class Choice internal constructor(val value: String) : LMStudioToolChoice

    @Serializable
    data class FunctionName(val name: String)

    @Serializable
    data class Function(val function: FunctionName) : LMStudioToolChoice {
        val type: String = "function"
    }

    companion object {
        // LMStudio api is too "dynamic", have to inline value here, so alas, no proper classes hierarchy, creating "objects" instead
        val Auto = Choice("auto")
        val Required = Choice("required")
        val None = Choice("none")
    }
}

@Serializable
internal data class LMStudioAudioConfig(
    val format: LMStudioAudioFormat,
    val voice: LMStudioAudioVoice,
)

@Serializable
internal enum class LMStudioAudioFormat {
    @SerialName("wav")
    WAV,
    @SerialName("pcm16")
    PCM16,
}

@Serializable
internal enum class LMStudioAudioVoice {
    @SerialName("alloy")
    Alloy,
}

@Serializable
internal data class LMStudioAudio(
    val data: String,
    val transcript: String? = null,
    val format: String? = null
)

internal object ContentSerializer : KSerializer<Content?> {
    @OptIn(InternalSerializationApi::class)
    override val descriptor: SerialDescriptor = buildSerialDescriptor("Content", PolymorphicKind.SEALED)

    override fun serialize(encoder: Encoder, value: Content?) {
        val jsonEncoder = encoder as JsonEncoder
        when (value) {
            null -> jsonEncoder.encodeNull()
            is Content.Text -> jsonEncoder.encodeString(value.value)
            is Content.Parts -> jsonEncoder.encodeSerializableValue(
                ListSerializer(ContentPart.serializer()),
                value.value
            )
        }
    }

    override fun deserialize(decoder: Decoder): Content? {
        val jsonDecoder = decoder as JsonDecoder
        val element = jsonDecoder.decodeJsonElement()

        return when (element) {
            is JsonNull -> null
            is JsonPrimitive -> Content.Text(element.content)
            is JsonArray -> Content.Parts(
                jsonDecoder.json.decodeFromJsonElement(
                    ListSerializer(ContentPart.serializer()),
                    element
                )
            )

            else -> throw SerializationException("Content must be either a string or an array")
        }
    }
}

@Serializable
internal data class LMStudioModelInfo(
    val id: String,
    @SerialName("object") val objectType: String,
    val type: String? = null,
    val publisher: String? = null,
    val arch: String? = null,
    @SerialName("compatibility_type") val compatibilityType: String? = null,
    val quantization: String? = null,
    val state: String? = null,
    @SerialName("max_context_length") val maxContextLength: Int? = null,
)

@Serializable
internal data class LMStudioModelsListResponse(
    @SerialName("object") val objectType: String,
    val data: List<LMStudioModelInfo>,
)

// Add at the end of the file

/**
 * Represents the different types of events that can be emitted by the LMStudio streaming client.
 */
public sealed interface LMStudioStreamEvent {
    /**
     * A chunk of text content from the language model.
     * @property text The text content chunk.
     */
    public data class TextChunk(val text: String) : LMStudioStreamEvent

    /**
     * A chunk of a tool call's arguments from the language model.
     * LMStudio may stream tool calls, with the function name and ID potentially arriving
     * separately or together with the first chunk of arguments. Consumers might need to
     * aggregate argument chunks based on `toolCallId` and `functionName`.
     *
     * @property toolCallId The unique identifier for the tool call.
     * @property functionName The name of the function being called.
     * @property argumentsChunk A chunk of the JSON string representing the function arguments.
     * @property toolCallIndex The index of the tool call in the list of tool calls within a single delta, if multiple are present.
     */
    public data class ToolCallChunk(
        val toolCallId: String,
        val functionName: String,
        val argumentsChunk: String,
        val toolCallIndex: Int? = null // In case a single delta has multiple new tool_calls with argument chunks
    ) : LMStudioStreamEvent

    /**
     * Indicates the reason why the stream finished.
     * @property reason The finish reason (e.g., "stop", "length", "tool_calls").
     */
    public data class FinishReason(val reason: String) : LMStudioStreamEvent

    // Consider adding an ErrorEvent if errors specific to stream content parsing
    // are to be emitted as part of the flow rather than terminating it.
    // For now, errors will terminate the flow as per previous error handling.
}
