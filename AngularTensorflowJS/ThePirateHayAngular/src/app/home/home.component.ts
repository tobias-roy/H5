import { Component } from '@angular/core';
import {Router} from '@angular/router';

@Component({
  selector: 'app-home',
  imports: [],
  templateUrl: './home.component.html',
  standalone: true,
  styleUrl: './home.component.css'
})
export class HomeComponent {
  constructor(private router: Router) {}

  GoToLiveFeed() {
    this.router.navigate(['/livefeed']);
  }
}
