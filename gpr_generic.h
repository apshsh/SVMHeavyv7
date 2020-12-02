// NB: the gradient of a GP is a GP
// mean of gradient is obtained from dg
// variance of gradient is obtained from dcov


//
// Gaussian Process (GP) base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Currently this is basically a wrap-around for a LS-SVR with C mapped to
// 1/sigma for noise regularisation.  This is equivalent to the standard
// GP regressor assuming Gaussian measurement noise.  By default the zero 
// mean case is assumed (translates to fixed bias), but you can change this
// and it will work for the general case (variance adjusted as per:
//
// Bull: Convergence Rates of Efficient Global Optimisation
//
// For gradients: http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
//

#ifndef _gpr_generic_h
#define _gpr_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"
#include "lsv_generic.h"




class GPR_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(GPR_Generic &a, GPR_Generic &b);
inline GPR_Generic &setzero(GPR_Generic &a);

class GPR_Generic : public ML_Base_Deref
{
public:

    // Constructors, destructors, assignment etc..

    GPR_Generic();
    GPR_Generic(const GPR_Generic &src);
    GPR_Generic(const GPR_Generic &src, const ML_Base *srcx);
    GPR_Generic &operator=(const GPR_Generic &src) { assign(src); return *this; }
    virtual ~GPR_Generic() { return; }

    virtual int prealloc(int expectedN) svm_override;

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) svm_override;
    virtual void semicopy(const ML_Base &src) svm_override;
    virtual void qswapinternal(ML_Base &b) svm_override;

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const svm_override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const svm_override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const svm_override;
    virtual std::istream &inputstream(std::istream &input ) svm_override;

    virtual       ML_Base &getML     (void)       svm_override { return static_cast<      ML_Base &>(getGPR());      }
    virtual const ML_Base &getMLconst(void) const svm_override { return static_cast<const ML_Base &>(getGPRconst()); }

    // Information functions (training data):

    virtual int NNC(int d)    const svm_override { return Nnc(d+1);           }
    virtual int type(void)    const svm_override { return -1;                 }
    virtual int subtype(void) const svm_override { return -1;                 }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const svm_override { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const svm_override { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual double C(void)         const svm_override { return 1/dsigma;                }
    virtual double sigma(void)     const svm_override { return dsigma;                  }
    virtual double Cclass(int d)   const svm_override { (void) d; return 1.0;           }

    virtual const Vector<gentype>                &y          (void) const svm_override { return dy;                        }
    virtual const Vector<int>                    &d          (void) const svm_override { return xd;                        }
    virtual const Vector<double>                 &Cweight    (void) const svm_override { return dCweight;                  }
    virtual const Vector<double>                 &sigmaweight(void) const svm_override { return dsigmaweight;              }

    virtual void npCweight    (double **res, int *dim) const svm_override { ML_Base::npCweight    (res,dim); return; }
    virtual void npCweightfuzz(double **res, int *dim) const svm_override { ML_Base::npCweightfuzz(res,dim); return; }
    virtual void npsigmaweight(double **res, int *dim) const svm_override { ML_Base::npsigmaweight(res,dim); return; }
    virtual void npepsweight  (double **res, int *dim) const svm_override { ML_Base::npepsweight  (res,dim); return; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override;
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override;

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return ML_Base::addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override;
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override;

    virtual int removeTrainingVector(int i) svm_override { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) svm_override;
    virtual int removeTrainingVector(int i, int num) svm_override;

    virtual int sety(int                i, const gentype         &nv) svm_override {                           dy("&",i)       = nv; return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) svm_override { retVector<gentype> tmpva; dy("&",i,tmpva) = nv; return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<gentype> &nv) svm_override {                           dy              = nv; return getQ().sety(  nv); }

    virtual int sety(int                i, double                nv) svm_override {                           dy("&",i) = nv;                 return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) svm_override { retVector<gentype> tmpva; dy("&",i,tmpva).castassign(nv); return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<double> &nv) svm_override {                           dy.castassign(nv);              return getQ().sety(  nv); }

    virtual int sety(int                i, const Vector<double>          &nv) svm_override { int ires = getQ().sety(i,nv);                                 dy("&",i)       = getQconst().y()(i);       return ires; }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &nv) svm_override { int ires = getQ().sety(i,nv); retVector<gentype> tmpva,tmpvb; dy("&",i,tmpva) = getQconst().y()(i,tmpvb); return ires; }
    virtual int sety(                      const Vector<Vector<double> > &nv) svm_override { int ires = getQ().sety(  nv);                                 dy              = getQconst().y();          return ires; }

    virtual int sety(int                i, const d_anion         &nv) svm_override { int ires = getQ().sety(i,nv);                                 dy("&",i)       = getQconst().y()(i);       return ires; }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &nv) svm_override { int ires = getQ().sety(i,nv); retVector<gentype> tmpva,tmpvb; dy("&",i,tmpva) = getQconst().y()(i,tmpvb); return ires; }
    virtual int sety(                      const Vector<d_anion> &nv) svm_override { int ires = getQ().sety(  nv);                                 dy              = getQconst().y();          return ires; }

    virtual int setd(int                i, int                nd) svm_override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) svm_override;
    virtual int setd(                      const Vector<int> &nd) svm_override;

    virtual int setCweight(int i,                double nv               ) svm_override;
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) svm_override;
    virtual int setCweight(                      const Vector<double> &nv) svm_override;

    virtual int setCweightfuzz(int i,                double nv               ) svm_override { (void) i; (void) nv; throw("Weight fuzzing not available for gpr models"); return 1; }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) svm_override { (void) i; (void) nv; throw("Weight fuzzing not available for gpr models"); return 1; }
    virtual int setCweightfuzz(                      const Vector<double> &nv) svm_override {           (void) nv; throw("Weight fuzzing not available for gpr models"); return 1; }

    virtual int setsigmaweight(int i,                double nv               ) svm_override;
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) svm_override;
    virtual int setsigmaweight(                      const Vector<double> &nv) svm_override;

    virtual int scaleCweight    (double s) svm_override;
    virtual int scaleCweightfuzz(double s) svm_override { (void) s; throw("Weight fuzzing not available for gpr models"); return 1; }
    virtual int scalesigmaweight(double s) svm_override;

    virtual const gentype &y(int i) const svm_override { return ( i >= 0 ) ? y()(i) : getQconst().y(i); }

    // General modification and autoset functions

    virtual int setC    (double xC)             svm_override { return setsigma(1/xC);                          }
    virtual int setsigma(double xsigma)         svm_override { dsigma = xsigma; return getQ().setC(1/sigma()); }
    virtual int setCclass  (int d, double xC)   svm_override { (void) d; (void) xC; throw("Weight classing not available for gpr models"); return 1; }

    virtual int scale(double a);

    virtual ML_Base &operator*=(double sf) svm_override { scale(sf); return *this; }

    virtual int scaleby(double sf) svm_override { *this *= sf; return 1; }

    // Sampling mode

    virtual int isSampleMode(void) const svm_override { return sampleMode; }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1, int sampType = 0) svm_override;

    // Training functions:

    virtual int train(int &res) svm_override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) svm_override { return getQ().train(res,killSwitch); }

    // Evaluation Functions:

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const;

    // var and covar functions

    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const svm_override;

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const svm_override { return ML_Base::covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const svm_override;






    // ================================================================
    //     Common functions for all GPs
    // ================================================================

    virtual       GPR_Generic &getGPR     (void)       { return *this; }
    virtual const GPR_Generic &getGPRconst(void) const { return *this; }

    // LS-SVM <-> GPR Translations
    //
    // bias  (delta) is replaced by muBias (though note that zero mean == zero bias assumed unless otherwise set)
    // alpha (gamma) is replaced by muWeight
    //
    // sigma = 1/C

    virtual int setmuWeight(const Vector<gentype> &nv) { return getQQ().setgamma(nv); }
    virtual int setmuBias  (const gentype         &nv) { return getQQ().setdelta(nv); }

    virtual const Vector<gentype> &muWeight(void) const { return getQQconst().gamma(); }
    virtual const gentype         &muBias  (void) const { return getQQconst().delta(); }

    virtual int isZeromuBias(void) const { return getQQconst().isZerodelta(); }
    virtual int isVarmuBias (void) const { return getQQconst().isVardelta();  }

    virtual int setZeromuBias(void) { return getQQ().setZerodelta(); }
    virtual int setVarmuBias (void) { return getQQ().setVardelta();  }

    virtual const Matrix<double> &gprGp(void) { return getQQ().lsvGp(); }

    // Likelihood

    virtual double loglikelihood(void) const { return getQQconst().lsvloglikelihood(); }

    // Base-level stuff
    //
    // This is overloaded by children to return correct Q type

    virtual       LSV_Generic &getQQ(void)            { return QQ; }
    virtual const LSV_Generic &getQQconst(void) const { return QQ; }

    virtual       ML_Base &getQ(void)            svm_override { return static_cast<      ML_Base &>(getQQ());      }
    virtual const ML_Base &getQconst(void) const svm_override { return static_cast<const ML_Base &>(getQQconst()); }




    // Grid generation, available anywhere.  Returns number of samples

    static int genSampleGrid(Vector<SparseVector<gentype> > &res, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1);

private:

    LSV_Generic QQ;

    double dsigma;
    Vector<double> dsigmaweight;
    Vector<double> dCweight;
    Vector<gentype> dy;

    // Local copy of d.  "d" as passed into lsv_generic is 0/2, d kept here
    // maintains -1,+1.  These can then be used in EP to enforce inequality
    // constraints.

    Vector<int> xd;

    // class counts

    Vector<int> Nnc; // number of vectors in each class (-1,0,+1,+2)

    // sampleMode: 0 normal
    //             1 this is a sample, so all evaluations of cov return sigma
    //               and all evaluations of g(x) are drawn from the posterior 
    //               N(g(x),cov(x,x)), then added as training data.
    //             2 this is a pre-sample.  Some pre-calculation has been done,
    //               but sample not actually taken yet.

    int sampleMode;

    Matrix<double> presample_L;
    Vector<double> presample_m;
    Vector<int> presample_p;
    int presample_s;


    GPR_Generic *thisthis;
    GPR_Generic **thisthisthis;
};

inline void qswap(GPR_Generic &a, GPR_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline GPR_Generic &setzero(GPR_Generic &a)
{
    a.restart();

    return a;
}

inline void GPR_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    GPR_Generic &b = dynamic_cast<GPR_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(xd          ,b.xd          );
    qswap(Nnc         ,b.Nnc         );
    qswap(dsigma      ,b.dsigma      );
    qswap(dsigmaweight,b.dsigmaweight);
    qswap(dy          ,b.dy          );
    qswap(dCweight    ,b.dCweight    );
    qswap(sampleMode  ,b.sampleMode  );
    qswap(presample_L ,b.presample_L );
    qswap(presample_m ,b.presample_m );
    qswap(presample_p ,b.presample_p );
    qswap(presample_s ,b.presample_s );

    return;
}

inline void GPR_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const GPR_Generic &b = dynamic_cast<const GPR_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    xd           = b.xd;
    Nnc          = b.Nnc;
    dsigma       = b.dsigma;
    dsigmaweight = b.dsigmaweight;
    dy           = b.dy;
    dCweight     = b.dCweight;
    sampleMode   = b.sampleMode;
    presample_L  = b.presample_L;
    presample_m  = b.presample_m;
    presample_p  = b.presample_p;
    presample_s  = b.presample_s;

    return;
}

inline void GPR_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const GPR_Generic &src = dynamic_cast<const GPR_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    xd           = src.xd;
    Nnc          = src.Nnc;
    dsigma       = src.dsigma;
    dsigmaweight = src.dsigmaweight;
    dy           = src.dy;
    dCweight     = src.dCweight;
    sampleMode   = src.sampleMode;
    presample_L  = src.presample_L;
    presample_m  = src.presample_m;
    presample_p  = src.presample_p;
    presample_s  = src.presample_s;

    return;
}

#endif
