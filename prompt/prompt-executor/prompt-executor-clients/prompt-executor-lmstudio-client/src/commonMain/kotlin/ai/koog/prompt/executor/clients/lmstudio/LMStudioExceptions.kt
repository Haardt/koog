package ai.koog.prompt.executor.clients.lmstudio

/**
 * Base class for all exceptions specific to the LMStudio client.
 */
public open class LMStudioException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * For network or connection issues not directly tied to an HTTP API error response.
 */
public class LMStudioConnectionException(message: String, cause: Throwable? = null) : LMStudioException(message, cause)

/**
 * Base class for errors returned by the LMStudio API that include an HTTP status code.
 *
 * @param httpStatusCode The HTTP status code.
 * @param errorJson The raw JSON error response string from the API, if available.
 * @param message The primary error message.
 * @param cause The underlying cause of this exception, if any.
 */
public open class LMStudioAPIException(
    public val httpStatusCode: Int,
    public val errorJson: String?,
    message: String,
    cause: Throwable? = null
) : LMStudioException(message, cause)

/**
 * For client-side errors (HTTP 4xx) from the LMStudio API.
 */
public open class LMStudioClientErrorException(
    httpStatusCode: Int,
    errorJson: String?,
    message: String,
    cause: Throwable? = null
) : LMStudioAPIException(httpStatusCode, errorJson, message, cause)

/**
 * Specifically for authentication failures (HTTP 401 or 403).
 */
public class LMStudioAuthenticationException(
    actualHttpStatusCode: Int, // Added parameter
    errorJson: String?,
    message: String = "Authentication failed with LMStudio API (Code: $actualHttpStatusCode)", // Updated default message
    cause: Throwable? = null
) : LMStudioClientErrorException(actualHttpStatusCode, errorJson, message, cause) // Pass actualHttpStatusCode

/**
 * Specifically for rate limiting errors (HTTP 429).
 *
 * @param retryAfterSeconds The number of seconds to wait before retrying, if provided by the API.
 */
public class LMStudioRateLimitException(
    errorJson: String?,
    message: String = "Rate limit exceeded with LMStudio API",
    public val retryAfterSeconds: Int? = null,
    cause: Throwable? = null
) : LMStudioClientErrorException(429, errorJson, message, cause)

/**
 * Specifically for resource not found errors (HTTP 404).
 */
public class LMStudioNotFoundException(
    errorJson: String?,
    message: String = "Resource not found on LMStudio API",
    cause: Throwable? = null
) : LMStudioClientErrorException(404, errorJson, message, cause)

/**
 * For server-side errors (HTTP 5xx) from the LMStudio API.
 */
public class LMStudioServerException(
    httpStatusCode: Int,
    errorJson: String?,
    message: String,
    cause: Throwable? = null
) : LMStudioAPIException(httpStatusCode, errorJson, message, cause)

/**
 * For issues encountered while parsing a successful (2xx) response from LMStudio,
 * such as unexpected JSON structure or missing required fields.
 */
public class LMStudioMalformedResponseException(message: String, cause: Throwable? = null) : LMStudioException(message, cause)

/**
 * For errors specific to SSE streaming after the connection has been established.
 */
public class LMStudioStreamException(message: String, cause: Throwable? = null) : LMStudioException(message, cause)

import kotlinx.serialization.Serializable

/**
 * Represents a common structure for error details returned by the LMStudio API in JSON format.
 * It attempts to accommodate various possible JSON error structures.
 */
@Serializable
public data class LMStudioErrorDetails(
    val error: ErrorObject? = null, // For nested error objects like OpenAI's
    val message: String? = null,    // For errors with message at root
    val type: String? = null,
    val code: String? = null // Or Int?, keep as String for flexibility
) {
    @Serializable
    public data class ErrorObject(
        val message: String,
        val type: String? = null,
        val param: String? = null,
        val code: String? = null // Or Int?
    )

    /**
     * Gets the most specific error message available.
     */
    public fun getDetailedMessage(): String? = error?.message ?: message

    /**
     * Gets the most specific error type available.
     */
    public fun getDetailedType(): String? = error?.type ?: type

    /**
     * Gets the most specific error code available.
     */
    public fun getDetailedCode(): String? = error?.code ?: code
}
