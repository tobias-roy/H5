import { Component, OnInit } from '@angular/core';
import * as cocoSSD from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import {Router} from '@angular/router';

@Component({
  selector: 'app-cam',
  templateUrl: './cam.component.html',
  standalone: true,
  styleUrls: ['./cam.component.css']
})
export class CamComponent implements OnInit {
  constructor(private router: Router) {
    console.log('CamComponent constructor');
  }
  public loading: boolean = true;
  videoStream: any;
  videoRef: any;
  modelCOCOSSD: any;

  async ngOnInit() {
    console.log('ngOnInit');
  }

  async ngAfterViewInit() {
    this.videoRef = document.getElementById('video');
    await this.startCamera();
    await tf.setBackend('webgl');
    await tf.ready();
    await this.loadModel();
  }

  async startCamera() {
    console.log('startCamera');
    navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    }).then(stream => {
      console.log('getUserMedia stream:', stream);
      this.videoStream = stream;
      if (this.videoRef != null) {
        this.videoRef.srcObject = this.videoStream;
        this.videoRef.onloadedmetadata = () => {
          console.log('videoRef onloadedmetadata');
          this.videoRef.play();
        };
      }
    })
  }

  async loadModel() {
    console.log('loadModel');
    this.modelCOCOSSD = await cocoSSD.load();
    console.log('Model loaded starting detection');
    this.loading = false;
    console.log('Loading set to false');
    this.detectFrame(this.videoRef, this.modelCOCOSSD);
  }

  detectFrame = async (video: any, model: any) => {
    console.log('detectFrame');
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      const predictions = await model.detect(video);
      console.log('Predictions:', predictions);
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    } else {
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    }
  }

  renderPredictions = (predictions: any[]) => {
    console.log('renderPredictions');
    const canvas = <HTMLCanvasElement>document.getElementById("canvas");
    const ctx = <CanvasRenderingContext2D>canvas.getContext("2d");

    canvas.width = 640;
    canvas.height = 480;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const width = prediction.bbox[2];
      const height = prediction.bbox[3];
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(prediction.class).width;
      const textHeight = parseInt(font, 10);
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    predictions.forEach(prediction => {
      const x = prediction.bbox[0];
      const y = prediction.bbox[1];
      const text = `${prediction.class} (${prediction.score.toFixed(2)})`;

      ctx.fillStyle = "#000000";
      ctx.fillText(text, x, y);
    });
  }

  GoToHome() {
    console.log('GoToHome');
    this.videoRef.pause();
    this.videoStream.getTracks().forEach((track: any) => {
      console.log('Stopping track:', track);
      track.stop();
    });
    this.router.navigate(['/']);
  }
}
