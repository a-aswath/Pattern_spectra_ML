/* mappers.h */
/* October 2000  Erik Urbach */

#ifndef MAPPERS_H
#define MAPPERS_H



int AreaMapper(double lambda, int num, double dlow, double dhigh);
int LinearMapper(double lambda, int num, double dlow, double dhigh);
int SqrtMapper(double lambda, int num, double dlow, double dhigh);
int Log2Mapper(double lambda, int num, double dlow, double dhigh);
int Log10Mapper(double lambda, int num, double dlow, double dhigh);



#endif /* MAPPERS_H */
