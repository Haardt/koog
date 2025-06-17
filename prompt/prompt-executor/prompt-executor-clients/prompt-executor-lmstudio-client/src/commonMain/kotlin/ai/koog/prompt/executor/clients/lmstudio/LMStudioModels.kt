package ai.koog.prompt.executor.clients.lmstudio

import ai.koog.prompt.executor.clients.LLModelDefinitions
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel

/**
 * Basic LM Studio models.
 */
public object LMStudioModels : LLModelDefinitions {
    /** Example model available in LM Studio. */
    public val Granite3BInstruct: LLModel = LLModel(
        provider = LLMProvider.LMStudio,
        id = "granite-3.0-2b-instruct",
        capabilities = listOf(
            LLMCapability.Completion,
            LLMCapability.ToolChoice,
            LLMCapability.Tools,
            LLMCapability.Temperature
        )
    )
}
