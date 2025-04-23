using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using OpenAI.RealtimeConversation;
using System.ClientModel;
using System.Text.Json;

namespace OpenAIRealtimeDemo
{

#pragma warning disable OPENAI002 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.
    public class Helpers
    {
        /// <summary>Helper method to get a RealtimeConversationClient for OpenAI.</summary>
        public static RealtimeConversationClient GetRealtimeConversationClient(string apiKey)
        {
            return new RealtimeConversationClient(
                model: "gpt-4o-realtime-preview",
                credential: new ApiKeyCredential(apiKey));
        }

        /// <summary>Helper method to get a RealtimeConversationClient for Azure OpenAI.</summary>
        public static RealtimeConversationClient GetAzureRealtimeConversationClient(string endpoint, string apiKey, string deploymentName)
        {
            // Use the AzureOpenAIClient to create a RealtimeConversationClient
            var aoaiClient = new Azure.AI.OpenAI.AzureOpenAIClient(
                new Uri(endpoint),
                new ApiKeyCredential(apiKey));

            return aoaiClient.GetRealtimeConversationClient(deploymentName);
        }

        /// <summary>Helper method to parse a function name for compatibility with Semantic Kernel plugins/functions.</summary>
        public static (string FunctionName, string? PluginName) ParseFunctionName(string fullyQualifiedName)
        {
            const string FunctionNameSeparator = "-";

            string? pluginName = null;
            string functionName = fullyQualifiedName;

            int separatorPos = fullyQualifiedName.IndexOf(FunctionNameSeparator, StringComparison.Ordinal);
            if (separatorPos >= 0)
            {
                pluginName = fullyQualifiedName.AsSpan(0, separatorPos).Trim().ToString();
                functionName = fullyQualifiedName.AsSpan(separatorPos + FunctionNameSeparator.Length).Trim().ToString();
            }

            return (functionName, pluginName);
        }

        /// <summary>Helper method to deserialize function arguments.</summary>
        public static KernelArguments? DeserializeArguments(string argumentsString)
        {
            var arguments = JsonSerializer.Deserialize<KernelArguments>(argumentsString);

            if (arguments is not null)
            {
                // Iterate over copy of the names to avoid mutating the dictionary while enumerating it
                var names = arguments.Names.ToArray();
                foreach (var name in names)
                {
                    arguments[name] = arguments[name]?.ToString();
                }
            }

            return arguments;
        }

        /// <summary>Helper method to process function result in order to provide it to the model as string.</summary>
        public static string? ProcessFunctionResult(object? functionResult)
        {
            if (functionResult is string stringResult)
            {
                return stringResult;
            }

            return JsonSerializer.Serialize(functionResult);
        }

        /// <summary>Helper method to convert Kernel plugins/function to realtime session conversation tools.</summary>
        public static IEnumerable<ConversationTool> ConvertFunctions(Kernel kernel)
        {
            foreach (var plugin in kernel.Plugins)
            {
                var functionsMetadata = plugin.GetFunctionsMetadata();

                foreach (var metadata in functionsMetadata)
                {
                    var toolDefinition = metadata.ToOpenAIFunction().ToFunctionDefinition(false);

                    yield return new ConversationFunctionTool(name: toolDefinition.FunctionName)
                    {
                        Description = toolDefinition.FunctionDescription,
                        Parameters = toolDefinition.FunctionParameters
                    };
                }
            }
        }
    }
}