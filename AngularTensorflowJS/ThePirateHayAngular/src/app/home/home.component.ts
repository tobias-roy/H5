import { Component } from '@angular/core';
import {Router} from '@angular/router';
import {NavigatorService} from '../navigator.service';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  standalone: true,
  styleUrl: './home.component.css'
})
export class HomeComponent {
  constructor(private router: Router, protected nav: NavigatorService) {}
}
