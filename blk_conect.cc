
//
// ML averaging block
//
// g(x) = mean(gi(x))
// gv(x) = mean(gv(x)) + var(gi(x))
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "blk_conect.h"
#include "gpr_generic.h"


BLK_Conect::BLK_Conect(int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    localygood = 0;

    locsampleMode = 0;
    locNsamp      = -1;
    locsampSplit  = 1;
    locsampType   = 0;

    setaltx(NULL);

    return;
}

BLK_Conect::BLK_Conect(const BLK_Conect &src, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(NULL);

    assign(src,0);

    return;
}

BLK_Conect::BLK_Conect(const BLK_Conect &src, const ML_Base *xsrc, int isIndPrune) : BLK_Generic(isIndPrune)
{
    thisthis = this;
    thisthisthis = &thisthis;

    setaltx(xsrc);

    assign(src,0);

    return;
}

BLK_Conect::~BLK_Conect()
{
    return;
}

std::ostream &BLK_Conect::printstream(std::ostream &output, int dep) const
{
    repPrint(output,'>',dep) << "ML Averaging block\n";

    return BLK_Generic::printstream(output,dep+1);
}

std::istream &BLK_Conect::inputstream(std::istream &input )
{
    return BLK_Generic::inputstream(input);
}























void BLK_Conect::fillCache(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).fillCache();
    }

    return;
}

int BLK_Conect::prealloc(int expectedN)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).prealloc(expectedN);
    }

    return 0;
}

int BLK_Conect::preallocsize(void) const
{
    return numReps() ? getRepConst(0).preallocsize() : 0;
}

void BLK_Conect::setmemsize(int memsize)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).setmemsize(memsize);
    }

    return;
}

void BLK_Conect::fudgeOn(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).fudgeOn();
    }

    return;
}

void BLK_Conect::fudgeOff(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).fudgeOff();
    }

    return;
}

void BLK_Conect::assumeConsistentX(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).assumeConsistentX();
    }

    return;
}

void BLK_Conect::assumeInconsistentX(void)
{
    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRep(ii).assumeInconsistentX();
    }

    return;
}

int BLK_Conect::addTrainingVector(int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return res;
}

int BLK_Conect::qaddTrainingVector(int i, const gentype &y, SparseVector<gentype> &x, double Cweigh, double epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return res;
}

int BLK_Conect::addTrainingVector(int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return res;
}

int BLK_Conect::qaddTrainingVector(int i, const Vector<gentype> &y, Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addTrainingVector(i,y,x,Cweigh,epsweigh);
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).removeTrainingVector(i);
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).removeTrainingVector(i,y,x);
    }

    return res;
}

int BLK_Conect::removeTrainingVector(int i, int num)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).removeTrainingVector(i,num);
    }

    return res;
}

int BLK_Conect::setx(int i, const SparseVector<gentype> &x)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setx(i,x);
    }

    return res;
}

int BLK_Conect::setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setx(i,x);
    }

    return res;
}

int BLK_Conect::setx(const Vector<SparseVector<gentype> > &x)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setx(x);
    }

    return res;
}

int BLK_Conect::sety(int i, const gentype &y)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,y);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<gentype> &y)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,y);
    }

    return res;
}

int BLK_Conect::sety(const Vector<gentype> &y)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(y);
    }

    return res;
}

int BLK_Conect::sety(int i, double z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<double> &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<double> &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::sety(int i, const Vector<double> &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<Vector<double> > &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<Vector<double> > &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::sety(int i, const d_anion &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<int> &i, const Vector<d_anion> &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(i,z);
    }

    return res;
}

int BLK_Conect::sety(const Vector<d_anion> &z)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).sety(z);
    }

    return res;
}

int BLK_Conect::setd(int i, int nd)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setd(i,nd);
    }

    return res;
}

int BLK_Conect::setd(const Vector<int> &i, const Vector<int> &nd)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setd(i,nd);
    }

    return res;
}

int BLK_Conect::setd(const Vector<int> &nd)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setd(nd);
    }

    return res;
}

int BLK_Conect::setCweight(int i, double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweight(i,nv);
    }

    return res;
}

int BLK_Conect::setCweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweight(i,nv);
    }

    return res;
}

int BLK_Conect::setCweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweight(nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(int i, double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweightfuzz(i,nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweightfuzz(i,nv);
    }

    return res;
}

int BLK_Conect::setCweightfuzz(const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCweightfuzz(nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(int i, double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setsigmaweight(i,nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setsigmaweight(i,nv);
    }

    return res;
}

int BLK_Conect::setsigmaweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setsigmaweight(nv);
    }

    return res;
}

int BLK_Conect::setepsweight(int i, double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setepsweight(i,nv);
    }

    return res;
}

int BLK_Conect::setepsweight(const Vector<int> &i, const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setepsweight(i,nv);;
    }

    return res;
}

int BLK_Conect::setepsweight(const Vector<double> &nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setepsweight(nv);;
    }

    return res;
}

int BLK_Conect::scaleCweight(double s)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).scaleCweight(s);
    }

    return res;
}

int BLK_Conect::scaleCweightfuzz(double s)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).scaleCweightfuzz(s);
    }

    return res;
}

int BLK_Conect::scalesigmaweight(double s)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).scalesigmaweight(s);
    }

    return res;
}

int BLK_Conect::scaleepsweight(double s)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).scaleepsweight(s);
    }

    return res;
}

int BLK_Conect::randomise(double sparsity)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).randomise(sparsity);
    }

    return res;
}

int BLK_Conect::autoen(void)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).autoen();
    }

    return res;
}

int BLK_Conect::renormalise(void)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).renormalise();
    }

    return res;
}

int BLK_Conect::realign(void)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).realign();
    }

    return res;
}

int BLK_Conect::setzerotol(double zt)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setzerotol(zt);
    }

    return res;
}

int BLK_Conect::setOpttol(double xopttol)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setOpttol(xopttol);
    }

    return res;
}

int BLK_Conect::setmaxitcnt(int xmaxitcnt)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setmaxitcnt(xmaxitcnt);
    }

    return res;
}

int BLK_Conect::setmaxtraintime(double xmaxtraintime)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setmaxtraintime(xmaxtraintime);
    }

    return res;
}

int BLK_Conect::setmaxitermvrank(int nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setmaxitermvrank(nv);
    }

    return res;
}

int BLK_Conect::setlrmvrank(double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setlrmvrank(nv);
    }

    return res;
}

int BLK_Conect::setztmvrank(double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setztmvrank(nv);
    }

    return res;
}

int BLK_Conect::setbetarank(double nv)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setbetarank(nv);
    }

    return res;
}

int BLK_Conect::setC(double xC)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setC(xC);
    }

    return res;
}

int BLK_Conect::setsigma(double xC)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setsigma(xC);
    }

    return res;
}

int BLK_Conect::seteps(double xC)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).seteps(xC);
    }

    return res;
}

int BLK_Conect::setCclass(int d, double xC)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setCclass(d,xC);
    }

    return res;
}

int BLK_Conect::setepsclass(int d, double xC)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setepsclass(d,xC);
    }

    return res;
}

int BLK_Conect::scale(double a)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).scale(a);
    }

    return res;
}

int BLK_Conect::reset(void)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).reset();
    }

    return res;
}

int BLK_Conect::restart(void)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).restart();
    }

    return res;
}

int BLK_Conect::home(void)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).home();
    }

    return res;
}

int BLK_Conect::settspaceDim(int newdim)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).settspaceDim(newdim);
    }

    return res;
}

int BLK_Conect::addtspaceFeat(int i)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addtspaceFeat(i);
    }

    return res;
}

int BLK_Conect::removetspaceFeat(int i)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).removetspaceFeat(i);
    }

    return res;
}

int BLK_Conect::addxspaceFeat(int i)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addxspaceFeat(i);
    }

    return res;
}

int BLK_Conect::removexspaceFeat(int i)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).removexspaceFeat(i);
    }

    return res;
}

int BLK_Conect::setsubtype(int i)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setsubtype(i);
    }

    return res;
}

int BLK_Conect::setorder(int neword)
{
    int ii;
    int res = 0;

    localygood = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).setorder(neword);
    }

    return res;
}

int BLK_Conect::addclass(int label, int epszero)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).addclass(label,epszero);
    }

    return res;
}

int BLK_Conect::train(int &res, svmvolatile int &killSwitch)
{
    int ii;
    int resi = 0;

    for ( ii = 0 ; ( ii < numReps() ) && !killSwitch ; ii++ )
    {
        resi |= getRep(ii).train(res,killSwitch);
    }

    return resi;
}

int BLK_Conect::disable(int i)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).disable(i);
    }

    return res;
}

int BLK_Conect::disable(const Vector<int> &i)
{
    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRep(ii).disable(i);
    }

    return res;
}


double BLK_Conect::sparlvl(void) const
{
    int ii;
    double res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res += (getRepConst(ii).sparlvl())/numReps();
    }

    return res;
}

const Vector<int> &BLK_Conect::d(void) const
{
    int ii,jj;
    ((**thisthisthis).dscratch).resize(N());
    ((**thisthisthis).dscratch) = 1;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        for ( jj = 0 ; jj < N() ; jj++ )
        {
            ((**thisthisthis).dscratch)("[]",jj) &= (getRepConst(ii).d())(jj);
        }
    }

    return dscratch;
}

const Vector<int> &BLK_Conect::alphaState(void) const
{
    int ii,jj;
    ((**thisthisthis).alphaStateScratch).resize(N());
    ((**thisthisthis).alphaStateScratch) = 1;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        for ( jj = 0 ; jj < N() ; jj++ )
        {
            ((**thisthisthis).alphaStateScratch)("[]",jj) &= (getRepConst(ii).alphaState())(jj);
        }
    }

    return alphaStateScratch;
}















const Vector<gentype> &BLK_Conect::y(void) const
{
    Vector<gentype> &res = (**thisthisthis).localy;

    if ( localygood <= 0 )
    {
        int ii,jj;

        Vector<Vector<gentype> > &yall = (**thisthisthis).localyparts;

        if ( !localygood )
        {
            yall.resize(numReps());

            int allowGridSample = 1;
            int dim = locxmin.size();
            int totsamp = allowGridSample ? ( dim ? (int) pow(locNsamp,dim) : 0 ) : locNsamp;

            NiceAssert( locxmax.size() == locxmin.size() );

            // Generate x grid only if required.

            Vector<SparseVector<gentype> > xgrid(totsamp);

            for ( ii = 0 ; ii < numReps() ; ii++ )
            {
                // This code is quite specific to globalopt.h
                //if ( isGPR(getRepConst(ii)) || isBLKConect(getRepConst(ii)) )
                if ( !( ( ( getRepConst(ii).type() >= 400 ) && ( getRepConst(ii).type() <= 499 ) ) || ( getRepConst(ii).type() == 212 ) || ( getRepConst(ii).type() == 215 ) ) )
                {
                    GPR_Generic::genSampleGrid(xgrid,locxmin,locxmax,locNsamp,locsampSplit);
                    break;
                }
            }

            for ( ii = 0 ; ii < numReps() ; ii++ )
            {
                // This code is quite specific to globalopt.h
                //if ( isGPR(getRepConst(ii)) || isBLKConect(getRepConst(ii)) )
                if ( ( ( getRepConst(ii).type() >= 400 ) && ( getRepConst(ii).type() <= 499 ) ) || ( getRepConst(ii).type() == 212 ) || ( getRepConst(ii).type() == 215 ) )
                {
                    yall("&",ii) = getRepConst(ii).y();
                    NiceAssert( yall(ii).size() == totsamp );
                }

                else
                {
                    for ( jj = 0 ; jj < totsamp ; jj++ )
                    {
                        getRepConst(ii).gg(yall("&",ii)("&",jj),xgrid(jj));
                    }
                }
            }
        }

        (**thisthisthis).localygood = 1;

        for ( ii = 0 ; ii < numReps() ; ii++ )
        {
            if ( !ii )
            {
                res = yall(ii);
                res.scale(getRepWeight(ii));
            }

            else
            {
                res.scaleAdd(getRepWeight(ii),yall(ii));
            }
        }
//errstream() << "phantomxy 0: " << Nmax << "," << locNsamp << "," << localy.size();
//if ( res.size() != 100 ) 
//{ 
//errstream() << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"; 
//for ( ii = 0 ; ii < numReps() ; ii++ )
//{
//errstream() << "phantomxyz " << ii << ": " << getRepConst(ii) << "\n";
//}
//}
//errstream() << "\n";
////errstream() << "phantomxyz localy = " << res << "\n";
    }

    return res;
}


int BLK_Conect::ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi) const
{
    Vector<gentype> vech(numReps());
    Vector<gentype> vecg(numReps());

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRepConst(ii).ghTrainingVector(vech("&",ii),vecg("&",ii),i,retaltg,pxyprodi);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecg("&",ii) *= getRepWeight(ii);

        if ( !isClassifier() )
        {
            vech("&",ii) *= getRepWeight(ii);
        }
    }

    if ( isClassifier() )
    {
        SparseVector<gentype> qq(vech);

        res |= combit.gh(resh,resg,qq,retaltg);
    }

    else
    {
        sum(resg,vecg);
        sum(resh,vech);
    }

    return res;
}

void BLK_Conect::fastg(gentype &res) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii));
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,xa,xb,xainfo,xbinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(gentype &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const
{
    Vector<gentype> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii));
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,xa,xb,xainfo,xbinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,xa,xb,xc,xainfo,xbinfo,xcinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,ib,ic,id,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}

void BLK_Conect::fastg(double &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).fastg(vecres("&",ii),ia,xa,xainfo);
    }

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        vecres("&",ii) *= getRepWeight(ii);
    }

    sum(res,vecres);

    return;
}




int BLK_Conect::gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg, const vecInfo *xinf, gentype ***pxyprodi) const
{
//errstream() << "phantomxgh gh 0: " << numReps() << "\n";
    Vector<gentype> vech(numReps());
    Vector<gentype> vecg(numReps());

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        res |= getRepConst(ii).gh(vech("&",ii),vecg("&",ii),x,retaltg,xinf,pxyprodi);
//errstream() << "phantomxgh gh(" << ii << ") 1: " << vech(ii) << "\n";
//errstream() << "phantomxgh gh(" << ii << ") 2: " << vecg(ii) << "\n";
//errstream() << "phantomxgh gh(" << ii << ") 2b: " << getRepConst(ii).type() << "\n";

        vecg("&",ii) *= getRepWeight(ii);
//errstream() << "phantomxgh gh(" << ii << ") 3: " << vecg(ii) << "\n";

        if ( !isClassifier() )
        {
            vech("&",ii) *= getRepWeight(ii);
//errstream() << "phantomxgh gh(" << ii << ") 4: " << vech(ii) << "\n";
        }
    }

    if ( isClassifier() )
    {
        SparseVector<gentype> qq(vech);

        res |= combit.gh(resh,resg,qq,retaltg);
//errstream() << "phantomxgh gh 5 WTF!!!\n";
    }

    else
    {
        sum(resg,vecg); // Not mean, this is mostly used by globalopt
//errstream() << "phantomxgh gh 6: " << resg << "\n";
        sum(resh,vech);
//errstream() << "phantomxgh gh 7: " << resh << "\n";
    }
//errstream() << "phantomxgh gh 8: " << res << "\n";

    return res;
}

void BLK_Conect::stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).stabProb(vecres("&",ii),x,p,pnrm,rot,mu,B);

        vecres("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res,vecres);
    }

    return;
}

void BLK_Conect::stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const
{
    Vector<double> vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).stabProbTrainingVector(vecres("&",ii),i,p,pnrm,rot,mu,B);

        vecres("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res,vecres);
    }

    return;
}


void BLK_Conect::dgTrainingVector(Vector<gentype> &res, gentype &resn, int i) const
{
    Vector<Vector<gentype> > vecres(numReps());
    Vector<gentype> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dgTrainingVector(vecres("&",ii),vecresn("&",ii),i);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dgTrainingVector(Vector<gentype> &res, const Vector<int> &i) const
{
    Vector<Vector<gentype> > vecres(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dgTrainingVector(vecres("&",ii),i);

        vecres("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(res,vecres);
    }

    return;
}

void BLK_Conect::dg(Vector<gentype> &res, gentype &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<gentype> > vecres(numReps());
    Vector<gentype> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dg(Vector<double> &res, double &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<double> > vecres(numReps());
    Vector<double> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<Vector<double> > > vecres(numReps());
    Vector<Vector<double> > vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

void BLK_Conect::dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<gentype> > vecresx(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dgX(vecresx("&",ii),x,xinf);

        vecresx("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(resx,vecresx);
    }

    return;
}

void BLK_Conect::dgX(Vector<double> &resx, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<double> > vecresx(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dgX(vecresx("&",ii),x,xinf);

        vecresx("&",ii).scale(getRepWeight(ii));
    }

    {
        sum(resx,vecresx);
    }

    return;
}

void BLK_Conect::dg(Vector<d_anion> &res, d_anion &resn, const SparseVector<gentype> &x, const vecInfo *xinf) const
{
    Vector<Vector<d_anion> > vecres(numReps());
    Vector<d_anion> vecresn(numReps());

    int ii;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        getRepConst(ii).dg(vecres("&",ii),vecresn("&",ii),x,xinf);

        vecres("&",ii).scale(getRepWeight(ii));
        vecresn("&",ii) *= getRepWeight(ii);
    }

    {
        sum(res, vecres );
        sum(resn,vecresn);
    }

    return;
}

int BLK_Conect::covTrainingVector(gentype &resv, gentype &resmu,int i, int j, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    Vector<gentype> vecv(numReps());
    Vector<gentype> vecg(numReps());
    //gentype dummy;

    (void) j;

    NiceAssert( i == j );

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        //res |= getRepConst(ii).ghTrainingVector(dummy,vecg("&",ii),i,0,pxyprodi);
        res |= getRepConst(ii).covTrainingVector(vecv("&",ii),vecg("&",ii),i,i,pxyprodi,pxyprodj,pxyprodij);

        vecg("&",ii) *= getRepWeight(ii);
        vecv("&",ii) *= getRepWeight(ii)*getRepWeight(ii);
    }

    gentype addterm;

    sum(resv,vecv);
    vari(addterm,vecg);

    addterm *= (double) numReps();
    resv += addterm;

    sum(resmu,vecg);

    return res;
}

int BLK_Conect::cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf, const vecInfo *xbinf, gentype ***pxyprodi, gentype ***pxyprodj, gentype **pxyprodij) const
{
    Vector<gentype> vecv(numReps());
    Vector<gentype> vecg(numReps());
    //gentype dummy;

    NiceAssert( xa == xb );

    int ii;
    int res = 0;

    for ( ii = 0 ; ii < numReps() ; ii++ )
    {
        //res = getRepConst(ii).gh(dummy,vecg("&",ii),xa,0,xainf,pxyprodi);
        res = getRepConst(ii).cov(vecv("&",ii),vecg("&",ii),xa,xb,xainf,xbinf,pxyprodi,pxyprodj,pxyprodij);

        vecg("&",ii) *= getRepWeight(ii);
        vecv("&",ii) *= getRepWeight(ii)*getRepWeight(ii);
    }

    gentype addterm;

    sum(resv,vecv);
    vari(addterm,vecg);

    addterm *= (double) numReps();
    resv += addterm;

    sum(resmu,vecg);

    return res;
}
