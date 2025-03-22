import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import { async } from 'rxjs';
import {resolve} from '@angular/compiler-cli';
tf.setBackend('webgl');
const threshold = 0.30;


@Component({
  selector: 'app-tfjsdetection',
  imports: [],
  templateUrl: './tfjsdetection.component.html',
  styleUrl: './tfjsdetection.component.css'
})
export class TfjsdetectionComponent implements OnInit {
    videoStream: any;
    videoRef: any;
    canvasRef: any;
    ngOnInit(): void {
      console.log('init');
    }

    async ngAfterViewInit() {
      this.videoRef = document.getElementById('videoFeed');
      this.canvasRef = document.getElementById('canvasOverlay');
      this.startCamera();
    }

    //Define the classes for detection
    classesDir = {
      1: {
        name: 'Aircraft carrier',
        id: 1
      },
      2: {
        name: 'Destroyer',
        id: 2
      },
      3: {
        name: 'Sailboat',
        id: 3
      },
      4: {
        name: 'Pirate boat',
        id: 4
      },
      5: {
        name: 'Bulker',
        id: 5
      },
      6: {
        name: 'Uboat',
        id: 6
      },
      7: {
        name: 'Container ship',
        id: 7
      },
      8: {
        name: 'Cruise ship',
        id: 8
      }
    }

    //In the funciton below using promises to do a Promise.all is the best approach
    async startCamera() {
      if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const webcamPromise = navigator.mediaDevices
          .getUserMedia({
            audio: false,
            video: {
                width: 640,
                height: 480
            }
          })
          .then(stream => {
            this.videoStream = stream;
            this.videoRef.srcObject = this.videoStream;
            return new Promise<void>((resolve, reject) => {
              this.videoRef.onloadedmetadata = () => {
                this.videoRef.play();
                resolve();
              };
            });
          });

        await tf.setBackend('webgl');
        await tf.ready();

        const modelPromise = loadModel();

        Promise.all([modelPromise, webcamPromise])
          .then(values => {
            this.detectFrame(this.videoRef, values[0]);
          }).catch(error => {
            console.error(error);
        })
      }
    }
  detectFrame = (video: any, model: any) => {
    tf.engine().startScope();
    model.executeAsync(process_input(video)).then((predictions: any) => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  }

  renderPredictions = (predictions: any) => {
    const ctx = this.canvasRef.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const boxes = predictions[4].arraySync();
    const scores = predictions[5].arraySync();
    const classes = predictions[6].dataSync();
    const detections = buildDetectedObjects(scores, threshold,
      boxes, classes, this.classesDir);
    console.log(detections);

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100*item["score"]).toFixed(2) + "%", x, y);
    });
  };
}

function process_input(video_frame: any){
  const tfImg = tf.browser.fromPixels(video_frame).toInt();
  return tfImg.transpose([0,1,2]).expandDims();
}

function buildDetectedObjects(scores: any, threshold: any, boxes: any, classes: any, classesDir: any) {
  const detectionObjects: any[] = [];
  const video_frame = document.getElementById('videoFeed');
  if (classes) {
    scores[0].forEach((score: any, i: number) => {
      if (score > threshold && video_frame != null) {
        const classId = classes[i];
        if (classesDir[classId]) {
          const bbox = [];
          const minY = boxes[0][i][0] * video_frame.offsetHeight;
          const minX = boxes[0][i][1] * video_frame.offsetWidth;
          const maxY = boxes[0][i][2] * video_frame.offsetHeight;
          const maxX = boxes[0][i][3] * video_frame.offsetWidth;
          bbox[0] = minX;
          bbox[1] = minY;
          bbox[2] = maxX - minX;
          bbox[3] = maxY - minY;
          detectionObjects.push({
            class: classId,
            label: classesDir[classId].name,
            score: score.toFixed(4),
            bbox: bbox
          });
        }
      }
    });
  }
  return detectionObjects;
}

async function loadModel() {
  let model: any;
  try {
    model = await loadGraphModel('https://raw.githubusercontent.com/tobias-roy/H5/refs/heads/MachineLearning/AngularTensorflowJS/model/ship-detector-resnet50/model.json', {onProgress: (number) => console.log(number)})
  } catch (error) {
    console.log('Error loading model:', error);
  }
  return model;
  }
