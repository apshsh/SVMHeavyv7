
//
// SVM base class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_generic_h
#define _svm_generic_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_base.h"


// NB: functions indented by 2 spaces should not be overloaded.


class SVM_Scalar;
class SVM_Binary;
template <class T> class SVM_Vector_redbin;
class SVM_Vector_atonce;
class SVM_Vector_Mredbin;
class SVM_Vector_Matonce;
template <class T> class SVM_Vector_atonce_temp;
class SVM_MultiC_redbin;
class SVM_MultiC_atonce;
class SVM_Generic;

// Swap and zeroing (restarting) functions

inline void qswap(SVM_Generic &a, SVM_Generic &b);
inline void qswap(SVM_Generic *&a, SVM_Generic *&b);

inline SVM_Generic &setzero(SVM_Generic &a);

class SVM_Generic : public ML_Base
{
public:

    friend class SVM_Scalar;
    friend class SVM_Binary;
    template <class T> friend class SVM_Vector_redbin;
    friend class SVM_Vector_atonce;
    friend class SVM_Vector_Mredbin;
    friend class SVM_Vector_Matonce;
    template <class T> friend class SVM_Vector_atonce_temp;
    friend class SVM_MultiC_atonce;
    friend class SVM_MultiC_redbin;

    // Constructors, destructors, assignment etc..

    SVM_Generic()                                            : ML_Base() { thisthis = this; thisthisthis = &thisthis; setaltx(NULL); gprevxvernum = -1; gprevgvernum = -1; gprevN = 0; gprevNb = 0;                return;       }
    SVM_Generic(const SVM_Generic &src)                      : ML_Base() { thisthis = this; thisthisthis = &thisthis; setaltx(NULL); gprevxvernum = -1; gprevgvernum = -1; gprevN = 0; gprevNb = 0; assign(src,0); return;       }
    SVM_Generic(const SVM_Generic &src, const ML_Base *srcx) : ML_Base() { thisthis = this; thisthisthis = &thisthis; setaltx(srcx); gprevxvernum = -1; gprevgvernum = -1; gprevN = 0; gprevNb = 0; assign(src,0); return;       }
    SVM_Generic &operator=(const SVM_Generic &src)                       {                                                                                                                          assign(src  ); return *this; }
    virtual ~SVM_Generic() { return; }

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize) { ML_Base::setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual int getparam (int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib) const;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const;

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input );

    virtual       ML_Base &getML     (void)       { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information functions (training data):

    virtual int N(void) const { return y().size(); }

    virtual int getInternalClass(const gentype &y) const;
    virtual int numInternalClasses(void)           const { return isanomalyOn() ? numClasses()+1 : numClasses(); }

    virtual double sparlvl(void) const { return N()-NNC(0) ? ((double) NZ()-NNC(0))/((double) N()-NNC(0)) : 1; }

    virtual const Vector<int> &alphaState (void) const { return xalphaState; }

    virtual int isClassifier(void) const { return 0; }
    virtual int isRegression(void) const { return 0; }

    // Kernel transfer

    virtual int isKVarianceNZ(void) const;

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const;
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const;

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const;
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const;
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const;

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int sety(int                i, const gentype         &y) { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<gentype> &y) { return ML_Base::sety(  y); }

    virtual int sety(int                i, double                y) { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<double> &y) { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<double> &y) { return ML_Base::sety(  y); }

    virtual int sety(int                i, const Vector<double>          &y) { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &y) { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<Vector<double> > &y) { return ML_Base::sety(  y); }

    virtual int sety(int                i, const d_anion         &y) { return ML_Base::sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &y) { return ML_Base::sety(i,y); }
    virtual int sety(                      const Vector<d_anion> &y) { return ML_Base::sety(  y); }

    virtual int setd(int                i, int                d) { (void) i; (void) d; throw("Function setd not available for this SVM type."); return 0; }
    virtual int setd(const Vector<int> &i, const Vector<int> &d) { (void) i; (void) d; throw("Function setd not available for this SVM type."); return 0; }
    virtual int setd(                      const Vector<int> &d) {           (void) d; throw("Function setd not available for this SVM type."); return 0; }

    // Evaluation Functions:

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const;
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const { ML_Base::dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const { ML_Base::dgTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const { ML_Base::dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const { ML_Base::dgTrainingVector(res,i); return; }


    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setAlpha(const Vector<gentype> &newAlpha);
    virtual int setBias (const gentype         &newBias );

    virtual int setAlphaR(const Vector<double>          &newAlpha) { (void) newAlpha; throw("Function setAlpha not available for this SVM type.");  return 0; }
    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) { (void) newAlpha; throw("Function setAlphaV not available for this SVM type."); return 0; }
    virtual int setAlphaA(const Vector<d_anion>         &newAlpha) { (void) newAlpha; throw("Function setAlphaA not available for this SVM type."); return 0; }

    virtual int setBiasR(const double         &newBias) { (void) newBias; throw("Function setBias not available for this SVM type.");  return 0; }
    virtual int setBiasV(const Vector<double> &newBias) { (void) newBias; throw("Function setBiasV not available for this SVM type."); return 0; }
    virtual int setBiasA(const d_anion        &newBias) { (void) newBias; throw("Function setBiasA not available for this SVM type."); return 0; }

    // Information functions (training data):
    //
    // NB: - it is important that isenabled() returns d(i).  This is used by
    //       errortest.h and is important for the scalar case.

    virtual int NZ (void)  const { return 0;               }
    virtual int NF (void)  const { return 0;               }
    virtual int NS (void)  const { return 0;               }
    virtual int NC (void)  const { return 0;               }
    virtual int NLB(void)  const { return 0;               }
    virtual int NLF(void)  const { return 0;               }
    virtual int NUF(void)  const { return 0;               }
    virtual int NUB(void)  const { return 0;               }
    virtual int NF (int q) const { (void) q; return 0;     }
    virtual int NZ (int q) const { (void) q; return 0;     }
    virtual int NS (int q) const { (void) q; return 0;     }
    virtual int NC (int q) const { (void) q; return 0;     }
    virtual int NLB(int q) const { (void) q; return 0;     }
    virtual int NLF(int q) const { (void) q; return 0;     }
    virtual int NUF(int q) const { (void) q; return 0;     }
    virtual int NUB(int q) const { (void) q; return 0;     }

    virtual const Vector<Vector<int> > &ClassRep(void)  const { const static Vector<Vector<int> > dummy; return dummy; }
    virtual int                         findID(int ref) const { (void) ref; return 0;                                  }

    virtual int isLinearCost(void)    const { return 0;           }
    virtual int isQuadraticCost(void) const { return 0;           }
    virtual int is1NormCost(void)     const { return 0;           }
    virtual int isVarBias(void)       const { return 0;           }
    virtual int isPosBias(void)       const { return 0;           }
    virtual int isNegBias(void)       const { return 0;           }
    virtual int isFixedBias(void)     const { return 0;           }
    virtual int isVarBias(int q)      const { (void) q; return 0; }
    virtual int isPosBias(int q)      const { (void) q; return 0; }
    virtual int isNegBias(int q)      const { (void) q; return 0; }
    virtual int isFixedBias(int q)    const { (void) q; return 0; }

    virtual int isNoMonotonicConstraints(void)    const { return 0; }
    virtual int isForcedMonotonicIncreasing(void) const { return 0; }
    virtual int isForcedMonotonicDecreasing(void) const { return 0; }

    virtual int isOptActive(void) const { return 0; }
    virtual int isOptSMO(void)    const { return 0; }
    virtual int isOptD2C(void)    const { return 0; }
    virtual int isOptGrad(void)   const { return 0; }

    virtual int isFixedTube(void)  const { return 0; }
    virtual int isShrinkTube(void) const { return 0; }

    virtual int isRestrictEpsPos(void) const { return 0; }
    virtual int isRestrictEpsNeg(void) const { return 0; }

    virtual int isClassifyViaSVR(void) const { return 0; }
    virtual int isClassifyViaSVM(void) const { return 0; }

    virtual int is1vsA(void)    const { return 0; }
    virtual int is1vs1(void)    const { return 0; }
    virtual int isDAGSVM(void)  const { return 0; }
    virtual int isMOC(void)     const { return 0; }
    virtual int ismaxwins(void) const { return 0; }
    virtual int isrecdiv(void)  const { return 0; }

    virtual int isatonce(void) const { return 0; }
    virtual int isredbin(void) const { return 0; }

    virtual int isKreal(void)   const { return 0; }
    virtual int isKunreal(void) const { return 0; }

    virtual int isanomalyOn(void)  const { return 0; }
    virtual int isanomalyOff(void) const { return 1; }

    virtual int isautosetOff(void)          const { return 0; }
    virtual int isautosetCscaled(void)      const { return 0; }
    virtual int isautosetCKmean(void)       const { return 0; }
    virtual int isautosetCKmedian(void)     const { return 0; }
    virtual int isautosetCNKmean(void)      const { return 0; }
    virtual int isautosetCNKmedian(void)    const { return 0; }
    virtual int isautosetLinBiasForce(void) const { return 0; }

    virtual double outerlr(void)       const { return MULTINORM_OUTERSTEP;     }
    virtual double outermom(void)      const { return MULTINORM_OUTERMOMENTUM; }
    virtual int    outermethod(void)   const { return MULTINORM_OUTERMETHOD;   }
    virtual double outertol(void)      const { return MULTINORM_OUTERSTEP;     }
    virtual double outerovsc(void)     const { return MULTINORM_OUTEROVSC;     }
    virtual int    outermaxitcnt(void) const { return MULTINORM_MAXITS;        }
    virtual int    outermaxcache(void) const { return MULTINORM_FULLCACHE_N;   }

    virtual       int      maxiterfuzzt(void) const { return DEFAULT_MAXITERFUZZT;                                 }
    virtual       int      usefuzzt(void)     const { return 0;                                                    }
    virtual       double   lrfuzzt(void)      const { return DEFAULT_LRFUZZT;                                      }
    virtual       double   ztfuzzt(void)      const { return DEFAULT_ZTFUZZT;                                      }
    virtual const gentype &costfnfuzzt(void)  const { const static gentype temp(DEFAULT_COSTFNFUZZT); return temp; }

    virtual int m(void) const { return DEFAULT_EMM; }

    virtual double LinBiasForce(void)   const { return 0;           }
    virtual double QuadBiasForce(void)  const { return 0;           }
    virtual double LinBiasForce(int q)  const { (void) q; return 0; }
    virtual double QuadBiasForce(int q) const { (void) q; return 0; }

    virtual double nu(void)     const { return 0; }
    virtual double nuQuad(void) const { return 0; }

    virtual double theta(void)   const { return 0; }
    virtual int    simnorm(void) const { return 0; }

    virtual double anomalyNu(void)    const { return 0; }
    virtual int    anomalyClass(void) const { return 0; }

    virtual double autosetCval(void)  const { return 0; }
    virtual double autosetnuval(void) const { return 0; }

    virtual int anomclass(void)          const { return +1; }
    virtual int singmethod(void)         const { return 0;  }
    virtual double rejectThreshold(void) const { return 0; }

    virtual const Matrix<double>          &Gp         (void)        const {             throw("Function Gp not available for this SVM type.");          const static Matrix<double> dummy;          return dummy;          }
    virtual const Matrix<double>          &XX         (void)        const {             throw("Function XX not available for this SVM type.");          const static Matrix<double> dummy;          return dummy;          }
    virtual const Vector<double>          &kerndiag   (void)        const {             throw("Function kerndiag not available for this SVM type.");    const static Vector<double> dummy;          return dummy;          }
    virtual const Vector<Vector<double> > &getu       (void)        const {             throw("Function getu not available for this SVM type.");        const static Vector<Vector<double> > dummy; return dummy;          }
    virtual const gentype                 &bias       (void)        const {                                                                                                                         return dbias;          }
    virtual const Vector<gentype>         &alpha      (void)        const {                                                                                                                         return dalpha;         }
    virtual const Vector<double>          &zR         (void)        const {             throw("Function zR not available for this SVM type.");          const static Vector<double> dummy;          return dummy;          }
    virtual const Vector<Vector<double> > &zV         (void)        const {             throw("Function zV not available for this SVM type.");          const static Vector<Vector<double> > dummy; return dummy;          }
    virtual const Vector<d_anion>         &zA         (void)        const {             throw("Function zA not available for this SVM type.");          const static Vector<d_anion> dummy;         return dummy;          }
    virtual const double                  &biasR      (void)        const {             throw("Function biasR not available for this SVM type.");       const static double dummy = 0.0;            return dummy;          }
    virtual const Vector<double>          &biasV      (int raw = 0) const { (void) raw; throw("Function biasV not available for this SVM type.");       const static Vector<double> dummy;          return dummy;          }
    virtual const d_anion                 &biasA      (void)        const {             throw("Function biasA not available for this SVM type.");                                                   return defaultanion(); }
    virtual const Vector<double>          &alphaR     (void)        const {             throw("Function alphaR not available for this SVM type.");      const static Vector<double> dummy;          return dummy;          }
    virtual const Vector<Vector<double> > &alphaV     (int raw = 0) const { (void) raw; throw("Function alphaV not available for this SVM type.");      const static Vector<Vector<double> > dummy; return dummy;          }
    virtual const Vector<d_anion>         &alphaA     (void)        const {             throw("Function alphaA not available for this SVM type.");      const static Vector<d_anion> dummy;         return dummy;          }

    virtual const double         &zR(int i) const { (void) i; throw("Function zR not available for this SVM type."); const static double dummy = 0;     return dummy; }
    virtual const Vector<double> &zV(int i) const { (void) i; throw("Function zV not available for this SVM type."); const static Vector<double> dummy; return dummy; }
    virtual const d_anion        &zA(int i) const { (void) i; throw("Function zA not available for this SVM type."); const static d_anion dummy(0.0);   return dummy; }

    // Training set modification:
    //
    // removeNonSupports: remove all non-support vectors from training set
    // trimTrainingSet: trim training set to desired size by removing smaller alphas

    virtual int removeNonSupports(void);
    virtual int trimTrainingSet(int maxsize);

    // General modification and autoset functions
    //
    // NB: - Class 0,-2,-3,... are reserved, all other classes may be used.
    //     - Class 0 means alpha = 0 (restricted) when using setd
    //     - Class 2 (in the regression context) means unconstrained (normal)
    //
    //
    // Autoset functions
    //
    // These functions tell the SVM to set various parameters automatically
    // based the given method.  They are persistent, in-so-far as adjusting
    // the kernel or adding/removing data will cause them to update the
    // parameters accordingly.  To turn off this feature, either use the
    // autosetOff() function (preferred) or call setC(getC()) (slow fallback
    // option).
    //
    // Cscaled:      C = Cval/N
    // CKmean:       C = mean(diag(G))
    // CKmedian:     C = median(diag(G))
    // CNKmean:      C = N*mean(diag(G))
    // CNKmedian:    C = N*median(diag(G))
    // LinBiasForce: C = Cval/(N*nuval),   eps = 0,   LinBiasForce = -Cval
    //
    // addclass: check if label is already present, and add if not
    //
    // NOTE: class 0,-2,-3,... are reserved, all other classes may be used.
    //       Class 0 means alpha = 0

    virtual int setLinearCost(void)                 {                           throw("Function setLinearCost not available for this SVM type.");    return 0; }
    virtual int setQuadraticCost(void)              {                           throw("Function setQuadraticCost not available for this SVM type."); return 0; }
    virtual int set1NormCost(void)                  {                           throw("Function set1NormCost not available for this SVM type.");     return 0; }
    virtual int setVarBias(void)                    {                           throw("Function setVarBias not available for this SVM type.");       return 0; }
    virtual int setPosBias(void)                    {                           throw("Function setPosBias not available for this SVM type.");       return 0; }
    virtual int setNegBias(void)                    {                           throw("Function setNegBias not available for this SVM type.");       return 0; }
    virtual int setFixedBias(double newbias)        { (void) newbias;           throw("Function setFixedBias not available for this SVM type.");     return 0; }
    virtual int setVarBias(int q)                   { (void) q;                 throw("Function setVarBias not available for this SVM type.");       return 0; }
    virtual int setPosBias(int q)                   { (void) q;                 throw("Function setPosBias not available for this SVM type.");       return 0; }
    virtual int setNegBias(int q)                   { (void) q;                 throw("Function setNegBias not available for this SVM type.");       return 0; }
    virtual int setFixedBias(int q, double newbias) { (void) q; (void) newbias; throw("Function setFixedBias not available for this SVM type.");     return 0; }
    virtual int setFixedBias(const gentype &newBias);

    virtual int setNoMonotonicConstraints(void)    { throw("Function setNoMonotonicConstraints not available for this SVM type.");    return 0; }
    virtual int setForcedMonotonicIncreasing(void) { throw("Function setForcedMonotonicIncreasing not available for this SVM type."); return 0; }
    virtual int setForcedMonotonicDecreasing(void) { throw("Function setForcedMonotonicDecreasing not available for this SVM type."); return 0; }

    virtual int setOptActive(void) { throw("Function setOptActive not available for this SVM type."); return 0; }
    virtual int setOptSMO(void)    { throw("Function setOptSMO not available for this SVM type.");    return 0; }
    virtual int setOptD2C(void)    { throw("Function setOptD2C not available for this SVM type.");    return 0; }
    virtual int setOptGrad(void)   { throw("Function setOptGrad not available for this SVM type.");   return 0; }

    virtual int setFixedTube(void)  { throw("Function setFixedTube not available for this SVM type.");  return 0; }
    virtual int setShrinkTube(void) { throw("Function setShrinkTube not available for this SVM type."); return 0; }

    virtual int setRestrictEpsPos(void) { throw("Function setRestrictEpsPos not available for this SVM type."); return 0; }
    virtual int setRestrictEpsNeg(void) { throw("Function setRestrictEpsNeg not available for this SVM type."); return 0; }

    virtual int setClassifyViaSVR(void) { throw("Function setClassifyViaSVR not available for this SVM type."); return 0; }
    virtual int setClassifyViaSVM(void) { throw("Function setClassifyViaSVM not available for this SVM type."); return 0; }

    virtual int set1vsA(void)    { throw("Function set1vsA not available for this SVM type.");    return 0; }
    virtual int set1vs1(void)    { throw("Function set1vs1 not available for this SVM type.");    return 0; }
    virtual int setDAGSVM(void)  { throw("Function setDAGSVM not available for this SVM type.");  return 0; }
    virtual int setMOC(void)     { throw("Function setMOC not available for this SVM type.");     return 0; }
    virtual int setmaxwins(void) { throw("Function setmaxwins not available for this SVM type."); return 0; }
    virtual int setrecdiv(void)  { throw("Function setrecdiv not available for this SVM type.");  return 0; }

    virtual int setatonce(void) { throw("Function setatonce not available for this SVM type."); return 0; }
    virtual int setredbin(void) { throw("Function setredbin not available for this SVM type."); return 0; }

    virtual int setKreal(void)   { throw("Function setKreal not available for this SVM type.");   return 0; }
    virtual int setKunreal(void) { throw("Function setKunreal not available for this SVM type."); return 0; }

    virtual int anomalyOn(int danomalyClass, double danomalyNu) { (void) danomalyClass; (void) danomalyNu; throw("Function anomalyOn not available for this SVM type.");  return 0; }
    virtual int anomalyOff(void)                                {                                          throw("Function anomalyOff not available for this SVM type."); return 0; }

    virtual int setouterlr(double xouterlr)           { (void) xouterlr;        throw("Function setouterlr not available for this SVM type.");       return 0; }
    virtual int setoutermom(double xoutermom)         { (void) xoutermom;       throw("Function setoutermom not available for this SVM type.");      return 0; }
    virtual int setoutermethod(int xoutermethod)      { (void) xoutermethod;    throw("Function setoutermethod not available for this SVM type.");   return 0; }
    virtual int setoutertol(double xoutertol)         { (void) xoutertol;       throw("Function setoutertol not available for this SVM type.");      return 0; }
    virtual int setouterovsc(double xouterovsc)       { (void) xouterovsc;      throw("Function setouterovsc not available for this SVM type.");     return 0; }
    virtual int setoutermaxitcnt(int xoutermaxits)    { (void) xoutermaxits;    throw("Function setoutermaxitcnt not available for this SVM type."); return 0; }
    virtual int setoutermaxcache(int xoutermaxcacheN) { (void) xoutermaxcacheN; throw("Function setoutermaxcache not available for this SVM type."); return 0; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)              { (void) xmaxiterfuzzt; throw("Function setmaxiterfuzzt not available for this SVM type."); return 0; }
    virtual int setusefuzzt(int xusefuzzt)                      { if ( xusefuzzt ) { throw("Function setusefuzzt not available for this SVM type."); }      return 0; }
    virtual int setlrfuzzt(double xlrfuzzt)                     { (void) xlrfuzzt;      throw("Function setlrfuzzt not available for this SVM type.");      return 0; }
    virtual int setztfuzzt(double xztfuzzt)                     { (void) xztfuzzt;      throw("Function setztfuzzt not available for this SVM type.");      return 0; }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)     { (void) xcostfnfuzzt;  throw("Function setcostfnfuzzt not available for this SVM type.");  return 0; }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt) { (void) xcostfnfuzzt;  throw("Function setcostfnfuzzt not available for this SVM type.");  return 0; }

    virtual int setm(int xm) { (void) xm; throw("Function setm not available for this SVM type."); return 0; }

    virtual int setLinBiasForce(double newval)         { (void) newval;           throw("Function setLinBiasForce not available for this SVM type.");  return 0; }
    virtual int setQuadBiasForce(double newval)        { (void) newval;           throw("Function setQuadBiasForce not available for this SVM type."); return 0; }
    virtual int setLinBiasForce(int q, double newval)  { (void) q; (void) newval; throw("Function setLinBiasForce not available for this SVM type.");  return 0; }
    virtual int setQuadBiasForce(int q, double newval) { (void) q; (void) newval; throw("Function setQuadBiasForce not available for this SVM type."); return 0; }

    virtual int setnu(double xnu)         { (void) xnu;     throw("Function setnu not available for this SVM type.");     return 0; }
    virtual int setnuQuad(double xnuQuad) { (void) xnuQuad; throw("Function setnuQuad not available for this SVM type."); return 0; }

    virtual int settheta(double nv) { (void) nv; throw("Function settheta not availble for this SVM type."); return 0; }
    virtual int setsimnorm(int nv)  { (void) nv; throw("Function setsimnorm not availble for this SVM type."); return 0; }

    virtual int autosetOff(void)                                     {                            throw("Function autosetOff not available for this SVM type.");          return 0; }
    virtual int autosetCscaled(double Cval)                          { (void) Cval;               throw("Function autosetCscaled not available for this SVM type.");      return 0; }
    virtual int autosetCKmean(void)                                  {                            throw("Function autosetCKmean not available for this SVM type.");       return 0; }
    virtual int autosetCKmedian(void)                                {                            throw("Function autosetCKmedian not available for this SVM type.");     return 0; }
    virtual int autosetCNKmean(void)                                 {                            throw("Function autosetCNKmean not available for this SVM type.");      return 0; }
    virtual int autosetCNKmedian(void)                               {                            throw("Function ausosetCNKmedian not available for this SVM type.");    return 0; }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { (void) nuval; (void) Cval; throw("Function autosetLinBiasForce not available for this SVM type."); return 0; }

    virtual void setanomalyclass(int n)        { (void) n;  throw("Function setanomalyclass not available for this SVM type.");    return; }
    virtual void setsingmethod(int nv)         { (void) nv; throw("Function setsingmethod not available for this SVM type.");      return; }
    virtual void setRejectThreshold(double nv) { (void) nv; throw("Function setRejectThreshold not available for this SVM type."); return; }

    // Evaluation Functions:
    //
    // NB: - g forms are specific to SVMs
    //     - return class (or sign), put unprocessed g(x) in res
    //     - Use raw != 0 to return non-reduced projection):

    virtual double quasiloglikelihood(void) const { return 1; }


protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual void basesetalpha(const Vector<gentype> &newalpha) { incgvernum(); dalpha = newalpha; return; }
    virtual void basesetbias (const gentype         &newbias ) { incgvernum(); dbias  = newbias;  return; }
    virtual void basesetbias (const double          &newbias ) { incgvernum(); dbias  = newbias;  return; }
    virtual void basesetbias (const d_anion         &newbias ) { incgvernum(); dbias  = newbias;  return; }
    virtual void basesetbias (const Vector<double>  &newbias ) { incgvernum(); dbias  = newbias;  return; }

    virtual void basesetalpha(int i, const gentype        &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i, const double         &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i, const d_anion        &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }
    virtual void basesetalpha(int i, const Vector<double> &newalpha) { incgvernum(); dalpha("&",i) = newalpha; return; }

    virtual void basescalealpha(double a) { incgvernum(); dalpha.scale(a); return; }
    virtual void basescalebias (double a) { incgvernum(); dbias *= a;      return; }

    virtual void basesetAlphaBiasFromAlphaBiasR(void);
    virtual void basesetAlphaBiasFromAlphaBiasV(void);
    virtual void basesetAlphaBiasFromAlphaBiasA(void);

    // General modification and autoset functions

//    virtual void setN (int newN) { (void) newN; throw("Function setN undefined for SVM.");  return; }

    // Don't use this!

    virtual int grablinbfq(void) const { return -1; }

    // Because LS-SVM needs to access this

    virtual const Vector<double> &diagoffset(void) const { throw("Function diagoffset not available for this SVM type.");  const static Vector<double> dummy; return dummy; }

    // Kernel cache selective access for gradient calculation

    virtual double getvalIfPresent(int numi, int numj, int &isgood) const { return ML_Base::getvalIfPresent(numi,numj,isgood); }

    // Inner-product cache: over-write this with a non-NULL return in classes where
    // a kernel cache is available

    virtual const Matrix<double> *getxymat(void) const { return NULL; }

    // Cached access to K matrix
public:
    virtual double getKval(int i, int j) const { (void) i; (void) j; throw("getKval not defined here"); return 0.0; }

    // Evaluation of 8x6 kernels

    virtual void fastg(gentype &res) const;
    virtual void fastg(gentype &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const;
    virtual void fastg(gentype &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;
    virtual void fastg(gentype &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const;
    virtual void fastg(gentype &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const;
    virtual void fastg(gentype &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const;

    virtual void fastg(double &res) const;
    virtual void fastg(double &res, int ia, const SparseVector<gentype> &xa, const vecInfo &xainfo) const;
    virtual void fastg(double &res, int ia, int ib, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo) const;
    virtual void fastg(double &res, int ia, int ib, int ic, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo) const;
    virtual void fastg(double &res, int ia, int ib, int ic, int id, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo) const;
    virtual void fastg(double &res, Vector<int> &ia, Vector<const SparseVector<gentype> *> &xa, Vector<const vecInfo *> &xainfo) const;

private:

    Vector<gentype> dalpha;
    gentype dbias;
    Vector<int> xalphaState;

    const gentype &alpha(int i, int j, const SparseVector<gentype> &dummy) const { (void) dummy; return alpha()(i)(j);                  }
    const double  &alpha(int i, int j, const SparseVector<double>  &dummy) const { (void) dummy; return (alpha()(i)(j)).cast_double(1); }
    const gentype &alpha(int i, int j, const Vector<gentype>       &dummy) const { (void) dummy; return alpha()(i)(j);                  }
    const double  &alpha(int i, int j, const Vector<double>        &dummy) const { (void) dummy; return (alpha()(i)(j)).cast_double(1); }

    // Cached stuff for accelerating evaluation of K2xfer

    SparseVector<SparseVector<gentype> > allxaprev;                     // previous xa vector (if different then need to recalculate)
    SparseVector<Matrix<SparseVector<gentype> > > allxadirectProdsFull; // pre-calculated direct products between xa, x(j) and x(k)
    SparseVector<SparseVector<gentype> > allxnormsgentype;              // cached evaluations of K(ia,ia), ia >= 0, if done
    SparseVector<SparseVector<double>  > allxnormsdouble;               // cached evaluations of K(ia,ia), ia >= 0, if done
    SparseVector<int> allprevxbvernum;                                  // version number of x

    Matrix<SparseVector<gentype> > gxvdirectProdsFull; // pre-calculated direct products between x(j) and x(k)
    Matrix<double> gaadirectProdsFull;                 // pre-calculated direct products between alpha(j) and alpha(k)
    Vector<int> gprevalphaState;                       // previous alpha state (element change used when updating above caches)
    Vector<int> gprevbalphaState;                      // like above, but used in different place

    int gprevxvernum; // version number of x for evaluator
    int gprevgvernum; // version number of alpha for evaluator
    int gprevN;       // N (most recent) for evaluator
    int gprevNb;      // like above, but used in different place

    // ...

    SVM_Generic *thisthis;
    SVM_Generic **thisthisthis;
};

inline void qswap(SVM_Generic &a, SVM_Generic &b)
{
    a.qswapinternal(b);

    return;
}

inline void qswap(SVM_Generic *&a, SVM_Generic *&b)
{
    SVM_Generic *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}

inline SVM_Generic *&setident (SVM_Generic *&a) { throw("Whatever"); return a; }
inline SVM_Generic *&setzero  (SVM_Generic *&a) { return a = NULL; }
inline SVM_Generic *&setposate(SVM_Generic *&a) { return a; }
inline SVM_Generic *&setnegate(SVM_Generic *&a) { throw("I reject your reality and substitute my own"); return a; }
inline SVM_Generic *&setconj  (SVM_Generic *&a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline SVM_Generic *&setrand  (SVM_Generic *&a) { throw("Blippity Blappity Blue"); return a; }
inline SVM_Generic *&postProInnerProd(SVM_Generic *&a) { return a; }

inline SVM_Generic &setzero(SVM_Generic &a)
{
    a.restart();

    return a;
}

inline void SVM_Generic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Generic &b = dynamic_cast<SVM_Generic &>(bb.getML());

    ML_Base::qswapinternal(b);

    qswap(dalpha     ,b.dalpha     );
    qswap(dbias      ,b.dbias      );
    qswap(xalphaState,b.xalphaState);

    qswap(allxaprev           ,b.allxaprev           );
    qswap(allxadirectProdsFull,b.allxadirectProdsFull);
    qswap(allxnormsgentype    ,b.allxnormsgentype    );
    qswap(allxnormsdouble     ,b.allxnormsdouble     );
    qswap(allprevxbvernum     ,b.allprevxbvernum     );

    qswap(gxvdirectProdsFull,b.gxvdirectProdsFull);
    qswap(gaadirectProdsFull,b.gaadirectProdsFull);
    qswap(gprevalphaState   ,b.gprevalphaState   );
    qswap(gprevbalphaState  ,b.gprevbalphaState  );

    qswap(gprevxvernum,b.gprevxvernum);
    qswap(gprevgvernum,b.gprevgvernum);
    qswap(gprevN      ,b.gprevN      );
    qswap(gprevNb     ,b.gprevNb     );

    return;
}

inline void SVM_Generic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Generic &b = dynamic_cast<const SVM_Generic &>(bb.getMLconst());

    ML_Base::semicopy(b);

    dalpha      = b.dalpha;
    dbias       = b.dbias;
    xalphaState = b.xalphaState;

    //allxaprev            = b.allxaprev;
    //allxadirectProdsFull = b.allxadirectProdsFull;
    //allxnormsgentype     = b.allxnormsgentype;
    //allxnormsdouble      = b.allxnormsdouble;
    //allprevxbvernum      = b.allprevxbvernum;

    //gxvdirectProdsFull = b.gxvdirectProdsFull;
    //gaadirectProdsFull = b.gaadirectProdsFull;
    //gprevalphaState    = b.gprevalphaState;
    //gprevbalphaState   = b.gprevbalphaState;

    //gprevxvernum = b.gprevxvernum;
    //gprevgvernum = b.gprevgvernum;
    //gprevN       = b.gprevN;
    //gprevNb      = b.gprevNb;

    incgvernum();

    return;
}

inline void SVM_Generic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Generic &src = dynamic_cast<const SVM_Generic &>(bb.getMLconst());

    ML_Base::assign(src,onlySemiCopy);

    dalpha      = src.dalpha;
    dbias       = src.dbias;
    xalphaState = src.xalphaState;

    allxaprev            = src.allxaprev;
    allxadirectProdsFull = src.allxadirectProdsFull;
    allxnormsgentype     = src.allxnormsgentype;
    allxnormsdouble      = src.allxnormsdouble;
    allprevxbvernum      = src.allprevxbvernum;

    gxvdirectProdsFull = src.gxvdirectProdsFull;
    gaadirectProdsFull = src.gaadirectProdsFull;
    gprevalphaState    = src.gprevalphaState;
    gprevbalphaState   = src.gprevbalphaState;

    gprevxvernum = src.gprevxvernum;
    gprevgvernum = src.gprevgvernum;
    gprevN       = src.gprevN;
    gprevNb      = src.gprevNb;

    return;
}






// This class allows for easy redirection.  Just inherit this instead of
// ML_Base and over-ride getQ and getQconst, and all functionality will bypass
// and go to wherever Q is.  You can then selectively modify the behaviour of
// member functions with appropriate over-riding.
//
// IMPORTANT: THIS IS NOT TO BE USED ON ITS OWN AS IT WILL INFINITELY RECURSE
//            UNLESS YOU OVER-RIDE getQ and getQconst!

/*

class ML_Base_Deref;

inline void qswap(ML_Base_Deref &a, ML_Base_Deref &b);
inline ML_Base_Deref &setzero(ML_Base_Deref &a);

class ML_Base_Deref : public ML_Base
{
public:

    virtual       ML_Base &getQ(void)            { return *this; }
    virtual const ML_Base &getQconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    ML_Base_Deref() : ML_Base() { return; }
    ML_Base_Deref(const ML_Base_Deref &src) : ML_Base() { assign(src,0); return; }
    ML_Base_Deref(const ML_Base_Deref &src, const ML_Base *srcx) : ML_Base() { setaltx(srcx); assign(src,0); return; }
    ML_Base_Deref &operator=(const ML_Base_Deref &src) { assign(src); return *this; }
    virtual ~ML_Base_Deref() { return; }

    virtual int prealloc(int expectedN)  svm_override { return getQ().prealloc(expectedN);  }
    virtual int preallocsize(void) const svm_override { return getQconst().preallocsize();  }
    virtual void setmemsize(int memsize) svm_override { getQ().setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) svm_override { getQ().assign(src,onlySemiCopy); return; }
    virtual void semicopy(const ML_Base &src)                     svm_override { getQ().semicopy(src);            return; }
    virtual void qswapinternal(ML_Base &b)                        svm_override { getQ().qswapinternal(b);         return; }

    virtual int getparam( int ind, gentype         &val, const gentype         &xa, int ia, const gentype         &xb, int ib) const svm_override { return getQconst().getparam( ind,val,xa,ia,xb,ib); }
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const svm_override { return getQconst().egetparam(ind,val,xa,ia,xb,ib); }

    virtual std::ostream &printstream(std::ostream &output, int dep) const svm_override { return getQconst().printstream(output,dep); }
    virtual std::istream &inputstream(std::istream &input          )       svm_override { return getQ().inputstream(input);           }

    virtual       ML_Base &getML     (void)       svm_override { return getQ().getML();           }
    virtual const ML_Base &getMLconst(void) const svm_override { return getQconst().getMLconst(); }

    // Information functions (training data):

    virtual int N(void)       const svm_override { return getQconst().N();       }
    virtual int NNC(int d)    const svm_override { return getQconst().NNC(d);    }
    virtual int type(void)    const svm_override { return getQconst().type();    }
    virtual int subtype(void) const svm_override { return getQconst().subtype(); }

    virtual int tspaceDim(void)    const svm_override { return getQconst().tspaceDim();    }
    virtual int xspaceDim(void)    const svm_override { return getQconst().xspaceDim();    }
    virtual int fspaceDim(void)    const svm_override { return getQconst().fspaceDim();    }
    virtual int tspaceSparse(void) const svm_override { return getQconst().tspaceSparse(); }
    virtual int xspaceSparse(void) const svm_override { return getQconst().xspaceSparse(); }
    virtual int numClasses(void)   const svm_override { return getQconst().numClasses();   }
    virtual int order(void)        const svm_override { return getQconst().order();        }

    virtual int isTrained(void) const svm_override { return getQconst().isTrained(); }
    virtual int isMutable(void) const svm_override { return getQconst().isMutable(); }
    virtual int isPool   (void) const svm_override { return getQconst().isPool   (); }

    virtual char gOutType(void) const svm_override { return getQconst().gOutType(); }
    virtual char hOutType(void) const svm_override { return getQconst().hOutType(); }
    virtual char targType(void) const svm_override { return getQconst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const svm_override { return getQconst().calcDist(ha,hb,ia,db); }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const svm_override { return getQconst().calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const svm_override { return getQconst().calcDistDbl(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const svm_override { return getQconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const svm_override { return getQconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const svm_override { return getQconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const svm_override { return getQconst().ClassLabels();        }
    virtual int getInternalClass(const gentype &y) const svm_override { return getQconst().getInternalClass(y);  }
    virtual int numInternalClasses(void)           const svm_override { return getQconst().numInternalClasses(); }
    virtual int isenabled(int i)                   const svm_override { return getQconst().isenabled(i);         }

    virtual const int *ClassLabelsInt(void) const svm_override { return getQconst().ClassLabelsInt();       }
    virtual int  getInternalClassInt(int y) const svm_override { return getQconst().getInternalClassInt(y); }

    virtual double C(void)         const svm_override { return getQconst().C();          }
    virtual double sigma(void)     const svm_override { return getQconst().sigma();     }
    virtual double eps(void)       const svm_override { return getQconst().eps();       }
    virtual double Cclass(int d)   const svm_override { return getQconst().Cclass(d);   }
    virtual double epsclass(int d) const svm_override { return getQconst().epsclass(d); }

    virtual int    memsize(void)      const svm_override { return getQconst().memsize();      }
    virtual double zerotol(void)      const svm_override { return getQconst().zerotol();      }
    virtual double Opttol(void)       const svm_override { return getQconst().Opttol();       }
    virtual int    maxitcnt(void)     const svm_override { return getQconst().maxitcnt();     }
    virtual double maxtraintime(void) const svm_override { return getQconst().maxtraintime(); }

    virtual int    maxitermvrank(void) const svm_override { return getQconst().maxitermvrank(); }
    virtual double lrmvrank(void)      const svm_override { return getQconst().lrmvrank();      }
    virtual double ztmvrank(void)      const svm_override { return getQconst().ztmvrank();      }

    virtual double betarank(void) const svm_override { return getQconst().betarank(); }

    virtual double sparlvl(void) const svm_override { return getQconst().sparlvl(); }

    virtual const Vector<SparseVector<gentype> > &x          (void) const svm_override { return getQconst().x();           }
    virtual const Vector<gentype>                &y          (void) const svm_override { return getQconst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const svm_override { return getQconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const svm_override { return getQconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const svm_override { return getQconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const svm_override { return getQconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const svm_override { return getQconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const svm_override { return getQconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const svm_override { return getQconst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const svm_override { return getQconst().alphaState();  }

    virtual void npCweight    (double **res, int *dim) const svm_override { getQconst().npCweight    (res,dim); return; }
    virtual void npCweightfuzz(double **res, int *dim) const svm_override { getQconst().npCweightfuzz(res,dim); return; }
    virtual void npsigmaweight(double **res, int *dim) const svm_override { getQconst().npsigmaweight(res,dim); return; }
    virtual void npepsweight  (double **res, int *dim) const svm_override { getQconst().npepsweight  (res,dim); return; }

    virtual int isClassifier(void) const svm_override { return getQconst().isClassifier(); }
    virtual int isRegression(void) const svm_override { return getQconst().isRegression(); }

    // Version numbers

    virtual int xvernum(void)        const svm_override { return getQconst().xvernum();        }
    virtual int xvernum(int altMLid) const svm_override { return getQconst().xvernum(altMLid); }
    virtual int incxvernum(void)           svm_override { return getQ().incxvernum();          }
    virtual int gvernum(void)        const svm_override { return getQconst().gvernum();        }
    virtual int gvernum(int altMLid) const svm_override { return getQconst().gvernum(altMLid); }
    virtual int incgvernum(void)           svm_override { return getQ().incgvernum();          }

    virtual int MLid(void) const svm_override { return getQconst().MLid(); }
    virtual int setMLid(int nv)  svm_override { return getQ().setMLid(nv); }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const svm_override { return getQconst().getaltML(res,altMLid); }

    // RKHS inner-product support

    virtual void mProdPt(double &res, int m, int *x) svm_override { getQ().mProdPt(res,m,x); return; }

    // Kernel Modification

    virtual const MercerKernel &getKernel(void) const                                           svm_override { return getQconst().getKernel();                              }
    virtual MercerKernel &getKernel_unsafe(void)                                                svm_override { return getQ().getKernel_unsafe();                            }
    virtual void prepareKernel(void)                                                            svm_override {        getQ().prepareKernel(); return;                       }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        svm_override { return getQ().resetKernel(modind,onlyChangeRowI,updateInfo); }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) svm_override { return getQ().setKernel(xkernel,modind,onlyChangeRowI);      }

    virtual void fillCache(void) svm_override { getQ().fillCache(); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) svm_override { getQ().K2bypass(nv); return; }

    virtual gentype &Keqn(gentype &res,                           int resmode = 1) const svm_override { return getQconst().Keqn(res,     resmode); }
    virtual gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const svm_override { return getQconst().Keqn(res,altK,resmode); }

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getQconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getQconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const svm_override { return getQconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const svm_override { return getQconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const svm_override { return getQconst().Km(res,xx); }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getQconst().K2ip(res,xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getQconst().distK(xa,xb,xainf,xbinf); }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const svm_override { return getQconst().phi2(res,ia,xa,xainf); }

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const svm_override { return getQconst().phi2(res,ia,xa,xainf); }

    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const svm_override { return getQconst().K0ip(res,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const svm_override { return  getQconst().K1ip(res,ia,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const svm_override { return  getQconst().K2ip(res,ia,ib,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const svm_override { return getQconst().K3ip(res,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const svm_override { return getQconst().K4ip(res,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const svm_override { return getQconst().Kmip(m,res,i,pxyprod,xx,xxinfo); }

    virtual double &K0ip(       double &res, const double &bias, const gentype **pxyprod = NULL) const svm_override { return getQconst().K0ip(res,bias,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const svm_override { return getQconst().K1ip(res,ia,bias,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const svm_override { return getQconst().K2ip(res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const svm_override { return getQconst().K3ip(res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const svm_override { return getQconst().K4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const svm_override { return getQconst().Kmip(m,res,i,bias,pxyprod,xx,xxinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(         res,altK,pxyprod,resmode); }
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getQconst().K0(order   ,res     ,pxyprod,resmode); }

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getQconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }

    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(         res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(         res,ia,ib,altK,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual double         &K2(              double         &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(spaceDim,res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getQconst().K2(order,   res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getQconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getQconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m         ,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m         ,res,i,bias,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m         ,res,i,altK,pxyprod,xx,xxinfo,resmode); }
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m         ,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m,spaceDim,res,i,pxyprod     ,xx,xxinfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getQconst().Km(m,order   ,res,i,pxyprod     ,xx,xxinfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const svm_override { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const svm_override { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual double distK(int i, int j) const svm_override { return getQconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const svm_override { return getQconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const svm_override { return getQconst().denseintK(res,i,j); }

    virtual void densedKdx(double &res, int i, int j, const double &bias) const svm_override { return getQconst().densedKdx(res,i,j,bias); }
    virtual void denseintK(double &res, int i, int j, const double &bias) const svm_override { return getQconst().denseintK(res,i,j,bias); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const svm_override { getQconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const svm_override { return getQconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const svm_override { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia,  int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const svm_override { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const svm_override { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm(gentype &res, int i, int j) const svm_override { return getQconst().xelm(res,i,j); }
    virtual int xindsize(int i) const svm_override { return getQconst().xindsize(i); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &y, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override { return  getQ().addTrainingVector(i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const gentype &y,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override { return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getQ().addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getQ().addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getQ().addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &y, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override { return  getQ().addTrainingVector(i,y,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &y,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override { return getQ().qaddTrainingVector(i,y,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i)                                       svm_override { return getQ().removeTrainingVector(i);     }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) svm_override { return getQ().removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                              svm_override { return getQ().removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) svm_override { return getQ().setx(i,x); }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) svm_override { return getQ().setx(i,x); }
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) svm_override { return getQ().setx(  x); }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) svm_override { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) svm_override { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) svm_override { return getQ().qswapx(  x,dontupdate); }

    virtual int sety(int                i, const gentype         &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<gentype> &nv) svm_override { return getQ().sety(  nv); }

    virtual int sety(int                i, double                nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<double> &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<double> &nv) svm_override { return getQ().sety(  nv); }

    virtual int sety(int                i, const Vector<double>          &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<Vector<double> > &nv) svm_override { return getQ().sety(  nv); }

    virtual int sety(int                i, const d_anion         &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &nv) svm_override { return getQ().sety(i,nv); }
    virtual int sety(                      const Vector<d_anion> &nv) svm_override { return getQ().sety(  nv); }

    virtual int setd(int                i, int                nd) svm_override { return getQ().setd(i,nd); }
    virtual int setd(const Vector<int> &i, const Vector<int> &nd) svm_override { return getQ().setd(i,nd); }
    virtual int setd(                      const Vector<int> &nd) svm_override { return getQ().setd(  nd); }

    virtual int setCweight(int i,                double nv               ) svm_override { return getQ().setCweight(i,nv); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getQ().setCweight(i,nv); }
    virtual int setCweight(                      const Vector<double> &nv) svm_override { return getQ().setCweight(  nv); }

    virtual int setCweightfuzz(int i,                double nv               ) svm_override { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) svm_override { return getQ().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(                      const Vector<double> &nv) svm_override { return getQ().setCweightfuzz(  nv); }

    virtual int setsigmaweight(int i,                double nv               ) svm_override { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getQ().setsigmaweight(i,nv); }
    virtual int setsigmaweight(                      const Vector<double> &nv) svm_override { return getQ().setsigmaweight(  nv); }

    virtual int setepsweight(int i,                double nv               ) svm_override { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getQ().setepsweight(i,nv); }
    virtual int setepsweight(                      const Vector<double> &nv) svm_override { return getQ().setepsweight(  nv); }

    virtual int scaleCweight    (double s) svm_override { return getQ().scaleCweight(s);     }
    virtual int scaleCweightfuzz(double s) svm_override { return getQ().scaleCweightfuzz(s); }
    virtual int scalesigmaweight(double s) svm_override { return getQ().scalesigmaweight(s); }
    virtual int scaleepsweight  (double s) svm_override { return getQ().scaleepsweight(s);   }

    virtual void assumeConsistentX  (void) svm_override { getQ().assumeConsistentX();   return; }
    virtual void assumeInconsistentX(void) svm_override { getQ().assumeInconsistentX(); return; }

    virtual int isXConsistent(void)        const svm_override { return getQconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const svm_override { return getQconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) svm_override { getQ().xferx(xsrc); return; }

    virtual const vecInfo &xinfo(int i)                       const svm_override { return getQconst().xinfo(i);                   }
    virtual int xtang(int i)                                  const svm_override { return getQconst().xtang(i);                   }
    virtual const SparseVector<gentype> &x(int i)             const svm_override { return getQconst().x(i);                       }
    virtual int xisrank(int i)                                const svm_override { return getQconst().xisrank(i);                 }
    virtual int xisgrad(int i)                                const svm_override { return getQconst().xisgrad(i);                 }
    virtual int xisrankorgrad(int i)                          const svm_override { return getQconst().xisrankorgrad(i);           }
    virtual int xisclass(int i, int defaultclass, int q = -1) const svm_override { return getQconst().xisclass(i,defaultclass,q); }
    virtual const gentype &y(int i)                           const svm_override { return getQconst().y(i);                       }

    // Basis stuff

    virtual int NbasisUU(void)    const svm_override { return getQconst().NbasisUU();    }
    virtual int basisTypeUU(void) const svm_override { return getQconst().basisTypeUU(); }
    virtual int defProjUU(void)   const svm_override { return getQconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const svm_override { return getQconst().VbasisUU(); }

    virtual int setBasisYUU(void)                     svm_override { return getQ().setBasisYUU();             }
    virtual int setBasisUUU(void)                     svm_override { return getQ().setBasisUUU();             }
    virtual int addToBasisUU(int i, const gentype &o) svm_override { return getQ().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU(int i)              svm_override { return getQ().removeFromBasisUU(i);      }
    virtual int setBasisUU(int i, const gentype &o)   svm_override { return getQ().setBasisUU(i,o);           }
    virtual int setBasisUU(const Vector<gentype> &o)  svm_override { return getQ().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)         svm_override { return getQ().setDefaultProjectionUU(d); }
    virtual int setBasisUU(int n, int d)              svm_override { return getQ().setBasisUU(n,d);           }

    virtual int NbasisVV(void)    const svm_override { return getQconst().NbasisVV();    }
    virtual int basisTypeVV(void) const svm_override { return getQconst().basisTypeVV(); }
    virtual int defProjVV(void)   const svm_override { return getQconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const svm_override { return getQconst().VbasisVV(); }

    virtual int setBasisYVV(void)                     svm_override { return getQ().setBasisYVV();             }
    virtual int setBasisUVV(void)                     svm_override { return getQ().setBasisUVV();             }
    virtual int addToBasisVV(int i, const gentype &o) svm_override { return getQ().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV(int i)              svm_override { return getQ().removeFromBasisVV(i);      }
    virtual int setBasisVV(int i, const gentype &o)   svm_override { return getQ().setBasisVV(i,o);           }
    virtual int setBasisVV(const Vector<gentype> &o)  svm_override { return getQ().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)         svm_override { return getQ().setDefaultProjectionVV(d); }
    virtual int setBasisVV(int n, int d)              svm_override { return getQ().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel(void) const                  svm_override { return getQconst().getUUOutputKernel();          }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       svm_override { return getQ().getUUOutputKernel_unsafe();        }
    virtual int resetUUOutputKernel(int modind = 1)                            svm_override { return getQ().resetUUOutputKernel(modind);       }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) svm_override { return getQ().setUUOutputKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) svm_override { return getQ().randomise(sparsity); }
    virtual int autoen(void)               svm_override { return getQ().autoen();            }
    virtual int renormalise(void)          svm_override { return getQ().renormalise();       }
    virtual int realign(void)              svm_override { return getQ().realign();           }

    virtual int setzerotol(double zt)                 svm_override { return getQ().setzerotol(zt);                 }
    virtual int setOpttol(double xopttol)             svm_override { return getQ().setOpttol(xopttol);             }
    virtual int setmaxitcnt(int xmaxitcnt)            svm_override { return getQ().setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) svm_override { return getQ().setmaxtraintime(xmaxtraintime); }

    virtual int setmaxitermvrank(int nv) svm_override { return getQ().setmaxitermvrank(nv); }
    virtual int setlrmvrank(double nv)   svm_override { return getQ().setlrmvrank(nv);      }
    virtual int setztmvrank(double nv)   svm_override { return getQ().setztmvrank(nv);      }

    virtual int setbetarank(double nv) svm_override { return getQ().setbetarank(nv); }

    virtual int setC    (double xC)             svm_override { return getQ().setC(xC);            }
    virtual int setsigma(double xsigma)         svm_override { return getQ().setsigma(xsigma);    }
    virtual int seteps  (double xeps)           svm_override { return getQ().seteps(xeps);        }
    virtual int setCclass  (int d, double xC)   svm_override { return getQ().setCclass(d,xC);     }
    virtual int setepsclass(int d, double xeps) svm_override { return getQ().setepsclass(d,xeps); }

    virtual int scale(double a) svm_override { return getQ().scale(a);  }
    virtual int reset(void)     svm_override { return getQ().reset();   }
    virtual int restart(void)   svm_override { return getQ().restart(); }
    virtual int home(void)      svm_override { return getQ().home();    }

    virtual ML_Base &operator*=(double sf) svm_override { return ( getQ() *= sf ); }

    virtual int scaleby(double sf) svm_override { return getQ().scaleby(sf); }

    virtual int settspaceDim(int newdim) svm_override { return getQ().settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     svm_override { return getQ().addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  svm_override { return getQ().removetspaceFeat(i);  }
    virtual int addxspaceFeat(int i)     svm_override { return getQ().addxspaceFeat(i);     }
    virtual int removexspaceFeat(int i)  svm_override { return getQ().removexspaceFeat(i);  }

    virtual int setsubtype(int i) svm_override { return getQ().setsubtype(i); }

    virtual int setorder(int neword)                 svm_override { return getQ().setorder(neword);        }
    virtual int addclass(int label, int epszero = 0) svm_override { return getQ().addclass(label,epszero); }

    // Sampling mode

    virtual int isSampleMode(void) const svm_override { return getQconst().isSampleMode(); }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1, int sampType = 0) svm_override { return getQ().setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType); }

    // Training functions:

    virtual void fudgeOn(void)  svm_override { getQ().fudgeOn();  return; }
    virtual void fudgeOff(void) svm_override { getQ().fudgeOff(); return; }

    virtual int train(int &res) svm_override { return getQ().train(res); }
    virtual int train(int &res, svmvolatile int &killSwitch) svm_override { return getQ().train(res,killSwitch); }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getQconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const svm_override { return getQconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getQconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual double eTrainingVector(int i) const svm_override { return getQconst().eTrainingVector(i); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const svm_override { return getQconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double         &dedgTrainingVector(double         &res, int i) const svm_override { return getQconst().dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const svm_override { return getQconst().dedgTrainingVector(res,i); }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const svm_override { return getQconst().dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const svm_override { return getQconst().dedgTrainingVector(res,i); }

    virtual double dedgAlphaTrainingVector(int i, int j) const svm_override { return getQconst().dedgAlphaTrainingVector(i,j); }
    virtual Matrix<double> &dedgAlphaTrainingVector(Matrix<double> &res) const svm_override { return getQconst().dedgAlphaTrainingVector(res); }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const svm_override { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const svm_override { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, int i) const svm_override { getQconst().deTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const svm_override { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const svm_override { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const svm_override { getQconst().deTrainingVectorX(resx,i); return; }

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const svm_override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const svm_override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const svm_override { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const svm_override { getQconst().dgTrainingVector(res,resn,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, gentype &resn, int i) const svm_override { getQconst().deTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const svm_override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const svm_override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const svm_override { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const svm_override { getQconst().dgTrainingVector(res,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const svm_override { getQconst().deTrainingVector(res,i); return; }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const svm_override { getQconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().gg(     resg,x,        xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().hh(resh,     x,        xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double e(const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { return getQconst().e(y,x,xinf); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const { return getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    virtual void dedg(double         &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(Vector<double> &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(d_anion        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(gentype        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dedg(res,y,x,xinf); return; }

    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dgX(resx,x,xinf); return; }
    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dgX(resx,x,xinf); return; }

    virtual void deX(Vector<gentype> &resx, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().deX(resx,y,x,xinf); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }

    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().dg(res,resn,x,xinf); return; }

    virtual void de(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getQconst().de(res,resn,y,x,xinf); return; }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const svm_override { getQconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const svm_override { return getQconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const svm_override { return getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const svm_override { return getQconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const svm_override { return getQconst().covar(resv,x); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const svm_override { return getQconst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const svm_override { return getQconst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const svm_override { return getQconst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const svm_override { return getQconst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) svm_override { getQ().setaltx(_altxsrc); return; }

    virtual int disable(int i)                svm_override { return getQ().disable(i); }
    virtual int disable(const Vector<int> &i) svm_override { return getQ().disable(i); }

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum   (SparseVector<gentype> &res) const svm_override { return getQconst().xsum(res);    }
    virtual const SparseVector<gentype> &xmean  (SparseVector<gentype> &res) const svm_override { return getQconst().xmean(res);   }
    virtual const SparseVector<gentype> &xmeansq(SparseVector<gentype> &res) const svm_override { return getQconst().xmeansq(res); }
    virtual const SparseVector<gentype> &xsqsum (SparseVector<gentype> &res) const svm_override { return getQconst().xsqsum(res);  }
    virtual const SparseVector<gentype> &xsqmean(SparseVector<gentype> &res) const svm_override { return getQconst().xsqmean(res); }
    virtual const SparseVector<gentype> &xmedian(SparseVector<gentype> &res) const svm_override { return getQconst().xmedian(res); }
    virtual const SparseVector<gentype> &xvar   (SparseVector<gentype> &res) const svm_override { return getQconst().xvar(res);    }
    virtual const SparseVector<gentype> &xstddev(SparseVector<gentype> &res) const svm_override { return getQconst().xstddev(res); }
    virtual const SparseVector<gentype> &xmax   (SparseVector<gentype> &res) const svm_override { return getQconst().xmax(res);    }
    virtual const SparseVector<gentype> &xmin   (SparseVector<gentype> &res) const svm_override { return getQconst().xmin(res);    }

    // Kernel normalisation function

    virtual int normKernelNone                  (void)                              svm_override { return getQ().normKernelNone();                                   }
    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) svm_override { return getQ().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) svm_override { return getQ().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) svm_override { return getQ().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const svm_override { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const svm_override { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const svm_override { return getQconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const svm_override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const svm_override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const svm_override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const svm_override { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const svm_override { return getQconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const svm_override { return getQconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const svm_override { return getQconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const svm_override { return getQconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const svm_override
    {
        return getQconst().detangle_x(i,usextang);
    }

    virtual int detangle_x(SparseVector<gentype> *&xuntang, vecInfo *&xinfountang,
                   const SparseVector<gentype> *&xnear, const SparseVector<gentype> *&xfar, const SparseVector<gentype> *&xfarfar, 
                   const vecInfo *&xnearinfo, const vecInfo *&xfarinfo, 
                   int &inear, int &ifar, const gentype *&ineartup, const gentype *&ifartup,
                   int &ilr, int &irr, int &igr, 
                   int &iokr, int &iok,
                   int i, int &idiagr, const SparseVector<gentype> *xx, const vecInfo *xxinfo, int &gradOrder, 
                   int &iplanr, int &iplan, int &iset, int usextang = 1, int allocxuntangifneeded = 1) const svm_override
    {
        return getQconst().detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,gradOrder,iplanr,iplan,iset,usextang,allocxuntangifneeded);
    }
};

inline void qswap(ML_Base_Deref &a, ML_Base_Deref &b)
{
    a.qswapinternal(b);

    return;
}

inline ML_Base_Deref &setzero(ML_Base_Deref &a)
{
    a.restart();

    return a;
}

*/

#endif
