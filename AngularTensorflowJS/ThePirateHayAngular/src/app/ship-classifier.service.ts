// src/app/services/ship-classifier.service.ts
import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ShipClassifierService {
  private mobilenet: any;
  private model: any;
  private CLASS_NAMES = ['Civil ships', 'Navy ships', 'Pirate ships'];
  private MOBILE_NET_INPUT_HEIGHT = 224;
  private MOBILE_NET_INPUT_WIDTH = 224;

  // Observables to track model state
  public modelLoaded$ = new BehaviorSubject<boolean>(false);
  public trainingProgress$ = new BehaviorSubject<number>(0);
  public currentPrediction$ = new BehaviorSubject<{label: string, confidence: number} | null>(null);

  constructor() {
    this.initMobilenet();
  }

  private async initMobilenet() {
    // Load the MobileNet model
    this.mobilenet = await tf.loadLayersModel(
      'models/initialModel.json'
    );

    // Get the layer we will use for feature extraction
    const layer = this.mobilenet.getLayer('conv_pw_13_relu');
    this.mobilenet = tf.model({inputs: this.mobilenet.inputs, outputs: layer.output});
  }

  // Train model using folder of images
  public async trainModelWithImages(imagesByClass: Map<number, File[]>) {
    const trainingDataInputs: any[] = [];
    const trainingDataOutputs: number[] = [];
    const examplesCount: {[key: number]: number} = {};

    // Process each class
    for (let classIndex = 0; classIndex < this.CLASS_NAMES.length; classIndex++) {
      examplesCount[classIndex] = 0;

      // Get images for this class
      const images = imagesByClass.get(classIndex) || [];

      // Process each image in this class
      for (const imageFile of images) {
        const img = await this.fileToImage(imageFile);

        // Extract features using MobileNet
        const imageFeatures = tf.tidy(() => {
          const imageTensor = tf.browser.fromPixels(img);
          const resizedTensorFrame = tf.image.resizeBilinear(
            imageTensor,
            [this.MOBILE_NET_INPUT_HEIGHT, this.MOBILE_NET_INPUT_WIDTH],
            true
          );
          const normalizedTensorFrame = resizedTensorFrame.div(255);
          return this.mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
        });

        trainingDataInputs.push(imageFeatures);
        trainingDataOutputs.push(classIndex);
        examplesCount[classIndex]++;

        // Update progress
        const totalProcessed = Object.values(examplesCount).reduce((a, b) => a + b, 0);
        const totalImages = Array.from(imagesByClass.values()).flat().length;
        this.trainingProgress$.next(totalProcessed / totalImages);
      }
    }

    await this.trainModel(trainingDataInputs, trainingDataOutputs);
    return examplesCount;
  }

  private async fileToImage(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  private async trainModel(trainingDataInputs: any[], trainingDataOutputs: number[]) {
    // Create and define model architecture
    const model = tf.sequential();
    model.add(tf.layers.dense({
      inputShape: [trainingDataInputs[0].shape[0]],
      units: 100,
      activation: 'relu'
    }));
    model.add(tf.layers.dense({
      units: this.CLASS_NAMES.length,
      activation: 'softmax'
    }));

    // Compile the model
    model.compile({
      optimizer: 'adam',
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });

    // Convert inputs and outputs to tensors
    const xs = tf.stack(trainingDataInputs);
    const ys = tf.tensor1d(trainingDataOutputs, 'int32');

    // Train the model
    await model.fit(xs, ys, {
      epochs: 10,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss = ${logs?.['loss']}, accuracy = ${logs?.['acc']}`);
          this.trainingProgress$.next((epoch + 1) / 10);
        }
      }
    });

    // Save the model and notify subscribers
    this.model = model;
    this.modelLoaded$.next(true);

    // Clean up tensors
    xs.dispose();
    ys.dispose();
  }

  // Predict from webcam or image
  public async predict(imageElement: HTMLImageElement | HTMLVideoElement): Promise<{label: string, confidence: number}> {
    if (!this.model || !this.mobilenet) {
      throw new Error('Model not loaded');
    }

    const result = tf.tidy(() => {
      // Preprocess the image
      const imgTensor = tf.browser.fromPixels(imageElement);
      const resized = tf.image.resizeBilinear(imgTensor,
        [this.MOBILE_NET_INPUT_HEIGHT, this.MOBILE_NET_INPUT_WIDTH], true);
      const normalized = resized.div(255);

      // Get MobileNet features
      const features = this.mobilenet.predict(normalized.expandDims());

      // Use our trained model to make a prediction
      return this.model!.predict(features) as tf.Tensor;
    });

    // Get the prediction results
    const predictions = await result.data();
    result.dispose();

    // Find the class with highest confidence
    let maxConfidence = 0;
    let predictedClass = 0;

    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] > maxConfidence) {
        maxConfidence = predictions[i];
        predictedClass = i;
      }
    }

    const prediction = {
      label: this.CLASS_NAMES[predictedClass],
      confidence: maxConfidence
    };

    // Update the current prediction observable
    this.currentPrediction$.next(prediction);

    return prediction;
  }

  // Save the model
  public async saveModel() {
    if (!this.model) {
      throw new Error('No model to save');
    }
    await this.model.save('localstorage://ship-classifier-model');
  }

  // Load a saved model
  public async loadSavedModel() {
    try {
      this.model = await tf.loadLayersModel('localstorage://ship-classifier-model');
      this.modelLoaded$.next(true);
      return true;
    } catch (error) {
      console.error('Failed to load model:', error);
      return false;
    }
  }
}
