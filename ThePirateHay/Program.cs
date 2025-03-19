using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace ThePirateHay
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Path to your existing ML.NET model
            string mlnetModelPath = "/Users/roy/Documents/Projects/MachineLearningH5/ThePirateHay/MLModel1.mlnet";
    
            // Output path for the ONNX model
            string onnxModelPath = "model.onnx";
    
            // Convert the model
            ModelConverter.ConvertModelToOnnx(mlnetModelPath, onnxModelPath);
    
            Console.WriteLine("\nAfter converting to ONNX, you can convert to TensorFlow.js format using the command:");
            Console.WriteLine("tensorflowjs_converter --input_format=onnx model.onnx ./tfjs_model");
    
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
        
        public static class ModelConverter
        {
            public static void ConvertModelToOnnx(string mlnetModelPath, string outputOnnxPath)
            {
                Console.WriteLine($"Starting conversion of {mlnetModelPath} to ONNX format...");
        
                try
                {
                    // Create MLContext
                    var mlContext = new MLContext();
            
                    // Load the existing ML.NET model
                    ITransformer loadedModel;
                    DataViewSchema inputSchema;
                    Console.WriteLine("Loading ML.NET model...");
                    using (var stream = new FileStream(mlnetModelPath, FileMode.Open))
                    {
                        loadedModel = mlContext.Model.Load(stream, out inputSchema);
                    }
                    Console.WriteLine("Model loaded successfully.");
            
                    // Create empty data view with the expected schema
                    // This is needed for the ONNX conversion process
                    Console.WriteLine("Creating sample data view with model schema...");
                    var emptyData = mlContext.Data.LoadFromEnumerable(
                        new List<object>(), 
                        inputSchema);
            
                    // Export the model to ONNX format
                    Console.WriteLine($"Converting model to ONNX format and saving to {outputOnnxPath}...");
                    using (var fileStream = new FileStream(outputOnnxPath, FileMode.Create))
                    {
                        mlContext.Model.ConvertToOnnx(loadedModel, emptyData, fileStream);
                    }
            
                    Console.WriteLine("Conversion completed successfully!");
                    Console.WriteLine($"ONNX model saved to: {outputOnnxPath}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during conversion: {ex.Message}");
                    Console.WriteLine(ex.StackTrace);
                }
            }
        }
    }
}
