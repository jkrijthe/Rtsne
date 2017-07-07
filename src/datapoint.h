#ifndef DATAPOINT_H
#define DATAPOINT_H

class DataPoint
{
  int _ind;
  
public:
  double* _x;
  int _D;
  DataPoint() {
    _D = 1;
    _ind = -1;
    _x = NULL;
  }
  DataPoint(int D, int ind, double* x) {
    _D = D;
    _ind = ind;
    _x = (double*) malloc(_D * sizeof(double));
    for(int d = 0; d < _D; d++) _x[d] = x[d];
  }
  DataPoint(const DataPoint& other) {                     // this makes a deep copy -- should not free anything
    if(this != &other) {
      _D = other.dimensionality();
      _ind = other.index();
      _x = (double*) malloc(_D * sizeof(double));      
      for(int d = 0; d < _D; d++) _x[d] = other.x(d);
    }
  }
  ~DataPoint() { if(_x != NULL) free(_x); }
  DataPoint& operator= (const DataPoint& other) {         // asignment should free old object
    if(this != &other) {
      if(_x != NULL) free(_x);
      _D = other.dimensionality();
      _ind = other.index();
      _x = (double*) malloc(_D * sizeof(double));
      for(int d = 0; d < _D; d++) _x[d] = other.x(d);
    }
    return *this;
  }
  int index() const { return _ind; }
  int dimensionality() const { return _D; }
  double x(int d) const { return _x[d]; }
};

#endif
