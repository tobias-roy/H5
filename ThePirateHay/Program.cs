using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace ThePirateHay
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Path to your existing ML.NET model
            string mlnetModelPath = "C/Users/tobia/source/repos/H5/ThePirateHay/MLModel1.mlnet";

            // Output path for the ONNX model
            string onnxModelPath = "model.onnx";

            MLModel1.Train("C:/Users/tobia/source/repos/H5/ThePirateHay/retrained.mlnet", onnxModelPath);
            //MLModel1.Train(mlnetModelPath, onnxModelPath);
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

                    // Print schema information
                    Console.WriteLine("Input schema columns:");
                    foreach (var column in inputSchema)
                    {
                        Console.WriteLine($"- {column.Name}: {column.Type}");
                    }

                    // Create a temporary dataset to use for prediction
                    byte[] dummyImageData = new byte[10 * 10 * 3]; // Small 10x10 RGB image
                    string tempImagePath = "dummy_image.jpg";
                    File.WriteAllBytes(tempImagePath, dummyImageData);

                    // Create some example data
                    var data = new List<ImageData>
                    {
                        new ImageData { ImagePath = tempImagePath }
                    };

                    // Load the data
                    var dataView = mlContext.Data.LoadFromEnumerable(data);

                    // Configure the prediction pipeline
                    var imageLoadingEstimator = mlContext.Transforms.LoadImages(
                        outputColumnName: "Image",
                        imageFolder: "",
                        inputColumnName: nameof(ImageData.ImagePath));

                    // Fit the image loading estimator
                    Console.WriteLine("Preparing transformation pipeline...");
                    var imageLoadingTransformer = imageLoadingEstimator.Fit(dataView);

                    // Apply the image loading transformer
                    var transformedData = imageLoadingTransformer.Transform(dataView);

                    // Create the final transformer by combining the image loading transformer and the loaded model
                    var finalPipeline = new TransformerChain<ITransformer>(
                        new ITransformer[] { imageLoadingTransformer, loadedModel });

                    // Export the model to ONNX format
                    Console.WriteLine($"Converting model to ONNX format and saving to {outputOnnxPath}...");
                    using (var fileStream = new FileStream(outputOnnxPath, FileMode.Create))
                    {
                        mlContext.Model.ConvertToOnnx(finalPipeline, transformedData, fileStream);
                    }

                    Console.WriteLine("Conversion completed successfully!");
                    Console.WriteLine($"ONNX model saved to: {outputOnnxPath}");

                    // Clean up the temporary file
                    if (File.Exists(tempImagePath))
                    {
                        File.Delete(tempImagePath);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during conversion: {ex.Message}");
                    Console.WriteLine(ex.StackTrace);
                }
            }

            // Simple class for image data input
            public class ImageData
            {
                public string ImagePath { get; set; }
            }

            // Output class for object detection
            public class ObjectDetectionOutput
            {
                public float[] BoundingBoxes { get; set; }
                public string[] Labels { get; set; }
                public float[] Scores { get; set; }
            }
        }
    }
}