import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfjsdetectionComponent } from './tfjsdetection.component';

describe('TfjsdetectionComponent', () => {
  let component: TfjsdetectionComponent;
  let fixture: ComponentFixture<TfjsdetectionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TfjsdetectionComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TfjsdetectionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
