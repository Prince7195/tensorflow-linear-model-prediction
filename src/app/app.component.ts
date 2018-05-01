import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  linearModel: tf.Sequential;
  pridiction: any;

  model: tf.Model;
  pridictions: any;

  ngOnInit() {
    this.trainNewModel();
  }

  async trainNewModel(): Promise<any> {
    // Defining a model for linear regression;
    this.linearModel = tf.sequential();
    this.linearModel.add(
      tf.layers.dense({units: 1, inputShape: [1]})
    );

    // Prepare model for training: Specify the loss and the optimizer
    this.linearModel.compile({
      loss: 'meanSquaredError',
      optimizer: 'sgd'
    });

    // Training data, completely random data
    const xs = tf.tensor1d([3.2, 4.4, 5.5, 6.7, 7.1, 9.7, 6.1, 7.5, 2.1]);
    const ys = tf.tensor1d([1.6, 2.7, 2.9, 3.1, 1.6, 2.5, 3.3, 2.5, 2.7]);

    // Train
    await this.linearModel.fit(xs, ys);

    console.log('Model Trained');

  }

  linearPrediction(val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.pridiction = Array.from(output.dataSync())[0];
  }

}
