import { Routes } from '@angular/router';
import {CamComponent} from './cam/cam.component';
import {HomeComponent} from './home/home.component';
import {TrainerComponent} from './trainer/trainer.component';
import {TfjsdetectionComponent} from './tfjsdetection/tfjsdetection.component';

export const routes: Routes = [
  { path: '', component: HomeComponent},
  { path: 'livefeed', component: CamComponent },
  { path: 'trainer', component: TrainerComponent},
  { path: 'detection', component: TfjsdetectionComponent}
];
