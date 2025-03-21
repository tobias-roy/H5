import { TestBed } from '@angular/core/testing';

import { ShipClassifierService } from './ship-classifier.service';

describe('ShipClassifierService', () => {
  let service: ShipClassifierService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ShipClassifierService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
