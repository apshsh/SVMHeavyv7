
//
// LS-SVM base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _lsv_generic_h
#define _lsv_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"
#include "svm_scalar.h"


class LSV_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(LSV_Generic &a, LSV_Generic &b);
inline LSV_Generic &setzero(LSV_Generic &a);

class LSV_Generic : public SVM_Scalar
{
public:

    // Constructors, destructors, assignment etc..

    LSV_Generic();
    LSV_Generic(const LSV_Generic &src);
    LSV_Generic(const LSV_Generic &src, const ML_Base *srcx);
    LSV_Generic &operator=(const LSV_Generic &src) { assign(src); return *this; }
    virtual ~LSV_Generic() { return; }

    virtual int prealloc(int expectedN) svm_override;
    virtual int preallocsize(void) const svm_override;
    virtual void setmemsize(int memsize) svm_override { SVM_Scalar::setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) svm_override;
    virtual void semicopy(const ML_Base &src) svm_override;
    virtual void qswapinternal(ML_Base &b) svm_override;

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const svm_override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const svm_override;

    virtual std::ostream &printstream(std::ostream &output, int dep) const svm_override;
    virtual std::istream &inputstream(std::istream &input ) svm_override;

    virtual       ML_Base &getML     (void)       svm_override { return static_cast<      ML_Base &>(getLSV     ()); }
    virtual const ML_Base &getMLconst(void) const svm_override { return static_cast<const ML_Base &>(getLSVconst()); }

    // Information functions (training data):

    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const svm_override;

    virtual int isUnderlyingScalar(void) const svm_override { return 0; }
    virtual int isUnderlyingVector(void) const svm_override { return 0; }
    virtual int isUnderlyingAnions(void) const svm_override { return 0; }

    virtual int getInternalClass(const gentype &y) const svm_override { return ML_Base::getInternalClass(y); }

    virtual double eps(void)       const svm_override { return 0.0;         }
    virtual double epsclass(int d) const svm_override { (void) d; return 1; }

    virtual const Vector<gentype> &y(void) const svm_override { return alltraintarg; }

    virtual int isClassifier(void) const svm_override { return 0; }
    virtual int isRegression(void) const svm_override { return 1; }

    // Kernel Modification

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

    virtual int sety(int                i, const gentype         &y) svm_override { SVM_Scalar::isStateOpt = 0; int res = ML_Base::sety(i,y);                           alltraintarg("&",i)       = y; return res; }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) svm_override { SVM_Scalar::isStateOpt = 0; int res = ML_Base::sety(i,y); retVector<gentype> tmpva; alltraintarg("&",i,tmpva) = y; return res; }
    virtual int sety(                      const Vector<gentype> &y) svm_override { SVM_Scalar::isStateOpt = 0; int res = ML_Base::sety(y);                             alltraintarg              = y; return res; }

    virtual int sety(int                i, double                y) svm_override { (void) i; (void) y; throw("sety fallback 1"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<double> &y) svm_override { (void) i; (void) y; throw("sety fallback 2"); return 1; }
    virtual int sety(                      const Vector<double> &y) svm_override {           (void) y; throw("sety fallback 3"); return 1; }

    virtual int sety(int                i, const Vector<double>          &y) svm_override { (void) i; (void) y; throw("Whatever"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) svm_override { (void) i; (void) y; throw("Whatever"); return 1; }
    virtual int sety(                      const Vector<Vector<double> > &y) svm_override {           (void) y; throw("Whatever"); return 1; }

    virtual int sety(int                i, const d_anion         &y) svm_override { (void) i; (void) y; throw("Whatever"); return 1; }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) svm_override { (void) i; (void) y; throw("Whatever"); return 1; }
    virtual int sety(                      const Vector<d_anion> &y) svm_override {           (void) y; throw("Whatever"); return 1; }

    virtual int setd(int                i, int                nd) svm_override;
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) svm_override;
    virtual int setd(                      const Vector<int> &nd) svm_override;

    virtual const gentype &y(int i) const svm_override { if ( i >= 0 ) { return y()(i); } return SVM_Scalar::y(i); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) svm_override { SVM_Scalar::isStateOpt = 0; return ML_Base::randomise(sparsity); }

    virtual int seteps(double xC) svm_override { NiceAssert( xC == 0 ); (void) xC; return 0; }

    virtual int scale(double a) svm_override;
    virtual int reset(void) svm_override;
    virtual int restart(void) svm_override { return ML_Base::restart(); }

    virtual int settspaceDim(int newdim) svm_override { return ML_Base::settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     svm_override { return ML_Base::addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  svm_override { return ML_Base::removetspaceFeat(i);  }

    virtual int setorder(int neword) svm_override { return ML_Base::setorder(neword); }

    // Training functions:

    virtual void fudgeOn(void)  svm_override { return; }
    virtual void fudgeOff(void) svm_override { return; }

    virtual int train(int &res) svm_override { svmvolatile int killSwitch = 0; return LSV_Generic::train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) svm_override;

    // Evaluation Functions:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return ML_Base::ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const svm_override { return ML_Base::covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double eTrainingVector(int i) const svm_override { return ML_Base::eTrainingVector(i); }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const svm_override;
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const svm_override;

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const svm_override;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const svm_override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const svm_override { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const svm_override { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const svm_override { ML_Base::dgTrainingVector(res,i); return; }





    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================
    //
    // Technically there are a bunch of SVM functions inheritted here.  I
    // probably should make them private, but that would be too much work.
    // Just don't use them.

    virtual       LSV_Generic &getLSV     (void)       { return *this; }
    virtual const LSV_Generic &getLSVconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setgamma(const Vector<gentype> &newgamma);
    virtual int setdelta(const gentype         &newdelta);

    // Additional information

    virtual int isVardelta (void) const { return SVM_Scalar::isVarBias();   }
    virtual int isZerodelta(void) const { return SVM_Scalar::isFixedBias(); }

    virtual const Vector<gentype> &gamma(void) const { return dalpha; }
    virtual const gentype         &delta(void) const { return dbias;  }

    virtual const Matrix<double> &lsvGp(void) const { return SVM_Scalar::Gp(); }

    // General modification and autoset functions

    virtual int setVardelta (void);
    virtual int setZerodelta(void);

    // Likelihood

    virtual double lsvloglikelihood(void) const { return SVM_Scalar::quasiloglikelihood(); }






    // ================================================================
    //     Required by K2xfer
    // ================================================================
    //
    // K2xfer requires these, though they aren't really part of LSV.

    virtual const gentype         &bias (void) const { return delta(); }
    virtual const Vector<gentype> &alpha(void) const { return gamma(); }





protected:

    // Variables

    Vector<gentype> dalpha;
    gentype dbias;

    // Targets

    Vector<gentype> alltraintarg;

    // The definition of zero depends on the target type

    virtual gentype &makezero(gentype &val) 
    { 
        val.force_null(); 

        return val; 
    }

    Vector<gentype> &makeveczero(Vector<gentype> &val) 
    { 
        int i; 

        if ( val.size() )
        {
            for ( i = 0 ; i < val.size() ; i++ )
            {
                makezero(val("&",i));
            }
        }

        return val; 
    }

    // If we need it

    LSV_Generic *thisthis;
    LSV_Generic **thisthisthis;
};

inline void qswap(LSV_Generic &a, LSV_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline LSV_Generic &setzero(LSV_Generic &a)
{
    a.restart();

    return a;
}

inline void LSV_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    LSV_Generic &b = dynamic_cast<LSV_Generic &>(bb.getML());

    SVM_Scalar::qswapinternal(b);

    qswap(dalpha      ,b.dalpha      );
    qswap(dbias       ,b.dbias       );
    qswap(alltraintarg,b.alltraintarg);

    return;
}

inline void LSV_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const LSV_Generic &b = dynamic_cast<const LSV_Generic &>(bb.getMLconst());

    SVM_Scalar::semicopy(b);

    dalpha       = b.dalpha;
    dbias        = b.dbias;
//    alltraintarg = b.alltraintarg;

    return;
}

inline void LSV_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const LSV_Generic &src = dynamic_cast<const LSV_Generic &>(bb.getMLconst());

    SVM_Scalar::assign(src,onlySemiCopy);

    dalpha       = src.dalpha;
    dbias        = src.dbias;
    alltraintarg = src.alltraintarg;

    return;
}

#endif
