namespace ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Spectre.Console;
using Console = Spectre.Console.AnsiConsole;

class Program
{
    static void Main(string[] args)
    {
        var ctx = new MLContext();
        try
        {
            ctx.Model.Load("model.zip", out var schema);
        }
        catch (Exception ex)
        {
            Console.WriteException(ex);
        }
        
        var dataView = ctx.Data.LoadFromTextFile<Models.SentimentData>("yelp.txt", hasHeader: false, separatorChar: '\t');
        var splitDataView = ctx.Data.TrainTestSplit(dataView, testFraction: 0.33);
        var estimator = ctx.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(Models.SentimentData.Text)
            )
            .Append(ctx.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName: "Features"));
        
        ITransformer model = default!;
        var rule = new Rule("Create and Train Model");
        Console.Live(rule).Start(console =>
        {
            //Model is training
            model = estimator.Fit(splitDataView.TrainSet);
            var predictions = model.Transform(splitDataView.TestSet);

            rule.Title = "Training Complete, Evaluating Accuracy.";
            console.Refresh();
            
            //Evaluate the accuracy
            var metrics = ctx.BinaryClassification.Evaluate(predictions);
            
            //Create table for the console
            var table = new Table()
                .MinimalBorder()
                .Title("Model Accuracy");
            table.AddColumns("Accuracy", "AreaUnderRoc", "F1Score");
            table.AddRow($"{metrics.Accuracy:P2}", $"{metrics.AreaUnderRocCurve:P2}", $"{metrics.F1Score:P2}");
            
            console.UpdateTarget(table);
            console.Refresh();
        });
        
        //Save the model we just trained
        ctx.Model.Save(model, dataView.Schema, "model.zip");

        //Loop over asking for user input
        while (true)
        {
            var text = AnsiConsole.Ask<string>("What's your [green]review text[/]?");
            var engine = ctx.Model.CreatePredictionEngine<Models.SentimentData, Models.SentimentPrediction>(model);

            var input = new Models.SentimentData { Text = text };
            var result = engine.Predict(input);
            var style = result.Prediction
                ? (color: "green", emoji: "👍")
                : (color: "red", emoji: "👎");
            Console.MarkupLine($"{style.emoji} [{style.color}]\"{text}\" ({result.Probability:P00})[/] ");
        }
    }
}