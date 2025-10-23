/* attrs.h */
/* October 2000  Erik Urbach */

#ifndef ATTRS_H
#define ATTRS_H



void *NewAreaData(int x, int y);
void DeleteAreaData(void *areaattr);
void AddToAreaData(void *areaattr, int x, int y);
void MergeAreaData(void *areaattr, void *childattr);
double AreaAttribute(void *areaattr);

void *NewEnclRectData(int x, int y);
void DeleteEnclRectData(void *rectattr);
void AddToEnclRectData(void *rectattr, int x, int y);
void MergeEnclRectData(void *rectattr, void *childattr);
double EnclRectAreaAttribute(void *rectattr);
double EnclRectDiagAttribute(void *rectattr);

void *NewPeriData(int x, int y);
void DeletePeriData(void *periattr);
void AddToPeriData(void *periattr, int x, int y);
void MergePeriData(void *periattr, void *childattr);
double PeriAreaAttribute(void *periattr);
double PeriPerimeterAttribute(void *periattr);
double PeriComplexityAttribute(void *periattr);
double PeriSimplicityAttribute(void *periattr);
double PeriCompactnessAttribute(void *periattr);

void *NewInertiaData(int x, int y);
void DeleteInertiaData(void *inertiaattr);
void AddToInertiaData(void *inertiaattr, int x, int y);
void MergeInertiaData(void *inertiaattr, void *childattr);
double InertiaAttribute(void *inertiaattr);
double InertiaDivA2Attribute(void *inertiaattr);

void *NewJaggedData(int x, int y);
void DeleteJaggedData(void *jaggedattr);
void AddToJaggedData(void *jaggedattr, int x, int y);
void MergeJaggedData(void *jaggedattr, void *childattr);
double JaggedAttribute(void *jaggedattr);
double JaggedCompactnessAttribute(void *jaggedattr);
double JaggedInertiaDivA2Attribute(void *jaggedattr);
double JaggednessAttribute(void *jaggedattr);

void *NewEntropyData(int x, int y);
void DeleteEntropyData(void *entropyattr);
void AddToEntropyData(void *entropyattr, int x, int y);
void MergeEntropyData(void *entropyattr, void *childattr);
double EntropyAttribute(void *entropyattr);

void *NewLambdamaxData(int x, int y);
void DeleteLambdamaxData(void *lambdaattr);
void AddToLambdamaxData(void *lambdaattr, int x, int y);
void MergeLambdamaxData(void *lambdaattr, void *childattr);
double LambdamaxAttribute(void *lambdaattr);

void *NewPosData(int x, int y);
void DeletePosData(void *posattr);
void AddToPosData(void *posattr, int x, int y);
void MergePosData(void *posattr, void *childattr);
double PosXAttribute(void *posattr);
double PosYAttribute(void *posattr);

void *NewLevelData(int x, int y);
void DeleteLevelData(void *levelattr);
void AddToLevelData(void *levelattr, int x, int y);
void MergeLevelData(void *levelattr, void *childattr);
double LevelAttribute(void *levelattr);

void *NewSumFluxData(int x, int y);
void DeleteSumFluxData(void *sumfluxattr);
void AddToSumFluxData(void *sumfluxattr, int x, int y);
void MergeSumFluxData(void *sumfluxattr, void *childattr);
double SumFluxAttribute(void *sumfluxattr);


#endif /* ATTRS_H */
