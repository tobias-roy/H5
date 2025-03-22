import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
tf.setBackend('webgl');
const threshold = 0.75;

@Component({
  selector: 'app-tfjsdetection',
  imports: [],
  templateUrl: './tfjsdetection.component.html',
  styleUrl: './tfjsdetection.component.css'
})
export class TfjsdetectionComponent implements OnInit {
    ngOnInit(): void {

    }
  async loadModel() {
    const model = await loadGraphModel('')
  }
}

