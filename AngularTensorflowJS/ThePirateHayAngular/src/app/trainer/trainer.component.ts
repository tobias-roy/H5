import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import {Router} from '@angular/router';
import {NavigatorService} from '../navigator.service';
import {NgStyle} from '@angular/common';
import {FormsModule} from '@angular/forms';

@Component({
  selector: 'app-trainer',
  imports: [
    NgStyle,
    FormsModule
  ],
  templateUrl: './trainer.component.html',
  standalone: true,
  styleUrl: './trainer.component.css'
})
export class TrainerComponent implements OnInit {
  constructor(private router: Router, protected nav: NavigatorService) {
  }

  async ngOnInit() {
    this.loadingModel = true;
    try{
        await this.loadModel();
    } catch (error){
      console.log('There was an error when initializing', error)
    }
  }

  //Constants used for training
  STATUS: string = '';
  CLASS_NAMES: string[] = ['Civil', 'Navy', 'Pirate'];
  MOBILE_NET_INPUT_WIDTH: number = 224;
  MOBILE_NET_INPUT_HEIGHT: number = 224;
  TRAINING_MODEL: any;

  //The initial model is Mobilenet_V3_Small_100_224
  modelPath: string = 'models/initialModel.json'
  loadingModel: boolean = false;
  loadingPercent: string = '0%';

  //Variable training inputs
  mobilenet: any; //For storing the model
  neurons: number = 128;

  trainingDataInputs: any[] = [];
  trainingDataOutputs: any[] = [];
  examplesCount: any[] = []; //keeps track of examples for each class
  predict: boolean = false; //When set to true predictions will start


  defineTrainingModel(){
    this.TRAINING_MODEL = tf.sequential()
    this.TRAINING_MODEL.add(tf.layers.dense({inputShape: [1024], units: Number(this.neurons), activation: 'relu'}))
    this.TRAINING_MODEL.add(tf.layers.dense({units: this.CLASS_NAMES.length, activation: 'softmax'}))
    this.TRAINING_MODEL.summary();
    this.TRAINING_MODEL.compile({
      optimizer: 'adam',
      loss: (this.CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  async loadModel() {
    try{
      this.mobilenet = await tf.loadGraphModel(this.modelPath, {onProgress: fraction => this.showProgress(fraction)});
      //Warm up the model
      tf.tidy(() => {
        let answer = this.mobilenet.predict(tf.zeros([1, this.MOBILE_NET_INPUT_HEIGHT, this.MOBILE_NET_INPUT_WIDTH, 3]))
        console.log(answer.shape);
      })
    } catch (error) {
      console.log('There was an error loading the model', error)
    }
    this.loadingModel = false;
  }

  showProgress(frac: number) {
    console.log(frac * 100);
    this.loadingPercent = `${frac * 100}%`;
  }

  async trainOnDataset() {

  }

}
