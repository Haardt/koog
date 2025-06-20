package ai.koog.prompt.executor.clients.lmstudio

import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation as ClientContentNegotiation
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNamingStrategy
import kotlinx.serialization.json.ClassDiscriminatorMode
import kotlinx.serialization.SerializationException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertNull

class LMStudioLLMClientTest {

    private val testModel = LLModel("test-model", LLMProvider.LMStudio, listOf(LLMCapability.Completion))
    private val testEmbeddingModel = LLModel("test-embedding-model", LLMProvider.LMStudio, listOf(LLMCapability.Embed))

    private fun createMockClient(
        httpStatusCode: HttpStatusCode,
        responseBody: String,
        responseHeaders: Map<String, String> = mapOf(HttpHeaders.ContentType to "application/json")
    ): LMStudioLLMClient {
        val mockEngine = MockEngine { _ ->
            respond(
                content = responseBody,
                status = httpStatusCode,
                headers = headersOf(responseHeaders.mapValues { listOf(it.value) })
            )
        }

        val clientJson = Json { // Ensure this Json config matches the one in LMStudioLLMClient
            ignoreUnknownKeys = true
            isLenient = true
            encodeDefaults = true
            explicitNulls = false
            namingStrategy = JsonNamingStrategy.SnakeCase
            classDiscriminatorMode = ClassDiscriminatorMode.NONE
        }
        val httpClient = HttpClient(mockEngine) {
            install(ClientContentNegotiation) {
                json(clientJson)
            }
        }
        return LMStudioLLMClient(settings = LMStudioClientSettings(), baseClient = httpClient)
    }

    // --- Error Handling Tests ---

    @Test
    fun `execute throws LMStudioAuthenticationException on 401`() = runBlocking {
        val errorJson = """{"error": {"message": "Auth failed", "type": "auth_error"}}"""
        val client = createMockClient(HttpStatusCode.Unauthorized, errorJson)

        val exception = assertFailsWith<LMStudioAuthenticationException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Auth failed", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(401, exception.httpStatusCode)
    }

    @Test
    fun `execute throws LMStudioAuthenticationException on 403`() = runBlocking {
        val errorJson = """{"error": {"message": "Forbidden", "type": "permission_error"}}"""
        val client = createMockClient(HttpStatusCode.Forbidden, errorJson)

        val exception = assertFailsWith<LMStudioAuthenticationException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Forbidden", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(403, exception.httpStatusCode)
    }

    @Test
    fun `execute throws LMStudioNotFoundException on 404`() = runBlocking {
        val errorJson = """{"error": {"message": "Not found", "type": "invalid_request_error"}}"""
        val client = createMockClient(HttpStatusCode.NotFound, errorJson)

        val exception = assertFailsWith<LMStudioNotFoundException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Not found", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(404, exception.httpStatusCode)
    }

    @Test
    fun `execute throws LMStudioRateLimitException on 429 with Retry-After header`() = runBlocking {
        val errorJson = """{"error": {"message": "Rate limit hit"}}"""
        val client = createMockClient(HttpStatusCode.TooManyRequests, errorJson, mapOf(
            HttpHeaders.ContentType to "application/json",
            HttpHeaders.RetryAfter to "60"
        ))

        val exception = assertFailsWith<LMStudioRateLimitException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Rate limit hit", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(429, exception.httpStatusCode)
        assertEquals(60, exception.retryAfterSeconds)
    }

    @Test
    fun `execute throws LMStudioRateLimitException on 429 without Retry-After header`() = runBlocking {
        val errorJson = """{"error": {"message": "Rate limit hit"}}"""
        val client = createMockClient(HttpStatusCode.TooManyRequests, errorJson)

        val exception = assertFailsWith<LMStudioRateLimitException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Rate limit hit", exception.getDetailedMessage())
        assertEquals(429, exception.httpStatusCode)
        assertNull(exception.retryAfterSeconds)
    }

    @Test
    fun `execute throws LMStudioClientErrorException on 400`() = runBlocking {
        val errorJson = """{"error": {"message": "Bad request"}}"""
        val client = createMockClient(HttpStatusCode.BadRequest, errorJson)

        val exception = assertFailsWith<LMStudioClientErrorException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Bad request", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(400, exception.httpStatusCode)
        assertTrue(exception !is LMStudioAuthenticationException)
        assertTrue(exception !is LMStudioNotFoundException)
        assertTrue(exception !is LMStudioRateLimitException)
    }

    @Test
    fun `execute throws LMStudioServerException on 500`() = runBlocking {
        val errorJson = """{"error": {"message": "Server issue"}}"""
        val client = createMockClient(HttpStatusCode.InternalServerError, errorJson)

        val exception = assertFailsWith<LMStudioServerException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Server issue", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(500, exception.httpStatusCode)
    }

    @Test
    fun `execute throws LMStudioAPIException on other unhandled API error codes like 501`() = runBlocking {
        val errorJson = """{"error": {"message": "Some other issue"}}"""
        val client = createMockClient(HttpStatusCode.NotImplemented, errorJson)

        val exception = assertFailsWith<LMStudioAPIException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals("Some other issue", exception.getDetailedMessage())
        assertEquals(errorJson, exception.errorJson)
        assertEquals(501, exception.httpStatusCode)
        assertTrue(exception !is LMStudioServerException || exception.httpStatusCode != 500)
    }

    @Test
    fun `execute handles non-JSON error response body`() = runBlocking {
        val errorText = "This is not JSON, just plain text error"
        val client = createMockClient(HttpStatusCode.BadRequest, errorText, mapOf(HttpHeaders.ContentType to "text/plain"))

        val exception = assertFailsWith<LMStudioClientErrorException> {
            client.execute(prompt { user("test") }, testModel)
        }
        assertEquals(errorText, exception.getDetailedMessage())
        assertEquals(errorText, exception.errorJson)
        assertEquals(400, exception.httpStatusCode)
    }

    @Test
    fun `execute throws LMStudioMalformedResponseException on successful response with unparsable JSON`() = runBlocking {
        val malformedSuccessJson = """{"id": "123", "object_type": "chat.completion", "created": 123, "model": "m", "choices": [ { "message": null } ]}"""
        val client = createMockClient(HttpStatusCode.OK, malformedSuccessJson)

        val exception = assertFailsWith<LMStudioMalformedResponseException> {
            client.execute(prompt { user("test") }, testModel)
        }
        val message = exception.message ?: ""
        val messageIndicatesParsingError = message.contains("Error parsing successful response") ||
                                           message.contains("Failed to process LMStudio response") ||
                                           message.contains("Empty choices in LMStudio response") ||
                                           message.contains("Content for the message is null") ||
                                           message.contains("Unexpected response from LMStudio: no tool calls and no content")


        assertTrue(messageIndicatesParsingError || exception.cause is SerializationException, "Exception message ('${'$'}{exception.message}') or cause ('${'$'}{exception.cause}') did not indicate a parsing/serialization error.")
    }
}
