//FIXME: consider moving "variants" to ml_base

//FIXME: add variants to return vectors where it sayd ADDHERE
//FIXME: ADDHERE

//Python examples:
//
//
// ML = ml_mutable.ML_Mutable(1)
// x = ml_mutable.new_doubleArray(2)
// doubleArray_setitem(x,0,1.23)
// doubleArray_setitem(x,1,2.91)
// y = x.gg(xx,2) const
// delete_doubleArray(a)
//
// NOTE: the dimension arument is always right after the pointer

//
// Mutable ML
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#ifndef _ml_mutable_h
#define _ml_mutable_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>

#include "ml_base.h"

#include "svm_generic.h"
#include "onn_generic.h"
#include "blk_generic.h"
#include "knn_generic.h"
#include "gpr_generic.h"
#include "lsv_generic.h"
#include "imp_generic.h"
#include "ssv_generic.h"
#include "mlm_generic.h"

#include "svm_single.h"
#include "svm_binary.h"
#include "svm_scalar.h"
#include "svm_multic.h"
#include "svm_vector.h"
#include "svm_anions.h"
#include "svm_autoen.h"
#include "svm_densit.h"
#include "svm_pfront.h"
#include "svm_biscor.h"
#include "svm_scscor.h"
#include "svm_gentyp.h"
#include "svm_planar.h"
#include "svm_mvrank.h"
#include "svm_mulbin.h"
#include "svm_cyclic.h"
#include "svm_simlrn.h"
#include "onn_scalar.h"
#include "onn_vector.h"
#include "onn_anions.h"
#include "onn_binary.h"
#include "onn_autoen.h"
#include "onn_gentyp.h"
#include "blk_nopnop.h"
#include "blk_consen.h"
#include "blk_avesca.h"
#include "blk_avevec.h"
#include "blk_aveani.h"
#include "blk_usrfna.h"
#include "blk_usrfnb.h"
#include "blk_userio.h"
#include "blk_calbak.h"
#include "blk_mexfna.h"
#include "blk_mexfnb.h"
#include "blk_mercer.h"
#include "blk_conect.h"
#include "blk_system.h"
#include "blk_kernel.h"
#include "blk_bernst.h"
#include "blk_batter.h"
#include "knn_densit.h"
#include "knn_binary.h"
#include "knn_multic.h"
#include "knn_gentyp.h"
#include "knn_scalar.h"
#include "knn_vector.h"
#include "knn_anions.h"
#include "knn_autoen.h"
#include "gpr_scalar.h"
#include "gpr_vector.h"
#include "gpr_anions.h"
#include "gpr_gentyp.h"
#include "gpr_binary.h"
#include "lsv_scalar.h"
#include "lsv_vector.h"
#include "lsv_anions.h"
#include "lsv_scscor.h"
#include "lsv_autoen.h"
#include "lsv_gentyp.h"
#include "lsv_planar.h"
#include "lsv_mvrank.h"
#include "imp_expect.h"
#include "imp_parsvm.h"
#include "ssv_scalar.h"
#include "ssv_binary.h"
#include "ssv_single.h"
#include "mlm_scalar.h"
#include "mlm_binary.h"
#include "mlm_vector.h"

#ifdef MAKENUMPYCOMP
#include "Python.h"
#include <numpy/numpyconfig.h>
#include <numpy/arrayobject.h>
#endif

class ML_Mutable;

// Swap function

#ifndef SWIG
inline void qswap(ML_Mutable &a, ML_Mutable &b);
inline void qswap(ML_Mutable *&a, ML_Mutable *&b);
#endif

#ifdef SWIG
class ML_Base;
#endif

class ML_Mutable : public ML_Base
{
public:

    // ================================================================
    //     Mutation functions
    // ================================================================
    //
    // setMLTypeMorph: set svm type and morph (ie transfer data).  This
    //                 is a lossy operation and may fail if old and new
    //                 types are incompatible.
    // setMLTypeClean: set svm type without data retention, so new type
    //                 is a clean start.

    virtual void setMLTypeMorph(int newmlType);
    virtual void setMLTypeClean(int newmlType);



    // ================================================================
    //     Common functions for all ML
    // ================================================================

    // Constructors, destructors, assignment etc..

    ML_Mutable();
    ML_Mutable(int type);
#ifndef SWIG
    ML_Mutable(const char *dummy, ML_Base *xtheML); // this is a special version that puts a wrapper around an existing ML.  Note that xtheML is not deleted when wrapper is!
    void settheMLdirect(ML_Base *xtheML);                // this is the corresponding "setter" version
    ML_Mutable(const ML_Mutable &src);
    ML_Mutable(const ML_Mutable &src, const ML_Base *srcx);
    ML_Mutable &operator=(const ML_Mutable &src) { assign(src); return *this; }
#endif
    virtual ~ML_Mutable();

    virtual int prealloc(int expectedN)  svm_override { return getML().prealloc(expectedN); }
    virtual int preallocsize(void) const svm_override { return getMLconst().preallocsize(); }
    virtual void setmemsize(int memsize) svm_override { getML().setmemsize(memsize); return; }
#ifndef SWIG
    virtual void assign(const ML_Base &src, int onlySemiCopy = 0) svm_override;
    virtual void semicopy(const ML_Base &src) svm_override;
    virtual void qswapinternal(ML_Base &b) svm_override;

    virtual int getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const svm_override;
    virtual int egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const svm_override;
#endif
    virtual std::ostream &printstream(std::ostream &output, int dep) const svm_override { return getMLconst().printstream(output,dep); }
    virtual std::istream &inputstream(std::istream &input ) svm_override;

    virtual       ML_Base &getML     (void)       svm_override { return (*(theML(mlind))).getML();      }
    virtual const ML_Base &getMLconst(void) const svm_override { return (*(theML(mlind))).getMLconst(); }

    // Information functions (training data):

    virtual int N(void)       const svm_override { return getMLconst().N  ();     }
    virtual int NNC(int d)    const svm_override { return getMLconst().NNC(d);    }
    virtual int type(void)    const svm_override { return getMLconst().type();    }
    virtual int subtype(void) const svm_override { return getMLconst().subtype(); }

    virtual int tspaceDim(void)    const svm_override { return getMLconst().tspaceDim();    }
    virtual int xspaceDim(void)    const svm_override { return getMLconst().xspaceDim();    }
    virtual int fspaceDim(void)    const svm_override { return getMLconst().fspaceDim();    }
    virtual int tspaceSparse(void) const svm_override { return getMLconst().tspaceSparse(); }
    virtual int xspaceSparse(void) const svm_override { return getMLconst().xspaceSparse(); }
    virtual int numClasses(void)   const svm_override { return getMLconst().numClasses();   }
    virtual int order(void)        const svm_override { return getMLconst().order();        }

    virtual int isTrained(void) const svm_override { return getMLconst().isTrained(); }
    virtual int isMutable(void) const svm_override { return 1;                        }
    virtual int isPool   (void) const svm_override { return 0;                        }

    virtual char gOutType(void) const svm_override { return getMLconst().gOutType(); }
    virtual char hOutType(void) const svm_override { return getMLconst().hOutType(); }
    virtual char targType(void) const svm_override { return getMLconst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const svm_override { return getMLconst().calcDist(ha,hb,ia,db); }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const svm_override { return getMLconst().calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const svm_override { return getMLconst().calcDistDbl(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const svm_override { return getMLconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const svm_override { return getMLconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const svm_override { return getMLconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const svm_override { return getMLconst().ClassLabels();               }
    virtual int getInternalClass(const gentype &y) const svm_override { return getMLconst().getInternalClass(y);         }
    virtual int numInternalClasses(void)           const svm_override { return getMLconst().numInternalClasses();        }
    virtual int isenabled(int i)                   const svm_override { return getMLconst().isenabled(i);                }

    virtual const int *ClassLabelsInt(void) const svm_override { return getMLconst().ClassLabelsInt();       }
    virtual int  getInternalClassInt(int y) const svm_override { return getMLconst().getInternalClassInt(y); }

    virtual double C(void)         const svm_override { return getMLconst().C();         }
    virtual double sigma(void)     const svm_override { return getMLconst().sigma();     }
    virtual double eps(void)       const svm_override { return getMLconst().eps();       }
    virtual double Cclass(int d)   const svm_override { return getMLconst().Cclass(d);   }
    virtual double epsclass(int d) const svm_override { return getMLconst().epsclass(d); }

    virtual int    memsize(void)      const svm_override { return getMLconst().memsize();      }
    virtual double zerotol(void)      const svm_override { return getMLconst().zerotol();      }
    virtual double Opttol(void)       const svm_override { return getMLconst().Opttol();       }
    virtual int    maxitcnt(void)     const svm_override { return getMLconst().maxitcnt();     }
    virtual double maxtraintime(void) const svm_override { return getMLconst().maxtraintime(); }

    virtual int    maxitermvrank(void) const svm_override { return getMLconst().maxitermvrank(); }
    virtual double lrmvrank(void)      const svm_override { return getMLconst().lrmvrank();      }
    virtual double ztmvrank(void)      const svm_override { return getMLconst().ztmvrank();      }

    virtual double betarank(void) const svm_override { return getMLconst().betarank(); }

    virtual double sparlvl(void) const svm_override { return getMLconst().sparlvl(); }
//FIXME: ADDHERE
    virtual const Vector<SparseVector<gentype> > &x          (void) const svm_override { return getMLconst().x();           }
    virtual const Vector<gentype>                &y          (void) const svm_override { return getMLconst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const svm_override { return getMLconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const svm_override { return getMLconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const svm_override { return getMLconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const svm_override { return getMLconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const svm_override { return getMLconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const svm_override { return getMLconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const svm_override { return getMLconst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const svm_override { return getMLconst().alphaState();  }

    virtual void npCweight    (double **res, int *dim) const svm_override { getMLconst().npCweight    (res,dim); return; }
    virtual void npCweightfuzz(double **res, int *dim) const svm_override { getMLconst().npCweightfuzz(res,dim); return; }
    virtual void npsigmaweight(double **res, int *dim) const svm_override { getMLconst().npsigmaweight(res,dim); return; }
    virtual void npepsweight  (double **res, int *dim) const svm_override { getMLconst().npepsweight  (res,dim); return; }

    virtual int isClassifier(void) const svm_override { return getMLconst().isClassifier(); }
    virtual int isRegression(void) const svm_override { return getMLconst().isRegression(); }

    // Version numbers

    virtual int xvernum(void)        const svm_override { return getMLconst().xvernum();        }
    virtual int xvernum(int altMLid) const svm_override { return getMLconst().xvernum(altMLid); }
    virtual int incxvernum(void)           svm_override { return getML().incxvernum();          }
    virtual int gvernum(void)        const svm_override { return getMLconst().gvernum();        }
    virtual int gvernum(int altMLid) const svm_override { return getMLconst().gvernum(altMLid); }
    virtual int incgvernum(void)           svm_override { return getML().incgvernum();          }

    virtual int MLid(void) const svm_override { return getMLconst().MLid(); }
    virtual int setMLid(int nv) svm_override { return getML().setMLid(nv); }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const svm_override { return getMLconst().getaltML(res,altMLid); }

    // RKHS inner-product support

    virtual void mProdPt(double &res, int m, int *x) svm_override { getML().mProdPt(res,m,x); return; }

    // Kernel Modification

    virtual const MercerKernel &getKernel(void) const                                           svm_override { return getMLconst().getKernel();                              }
    virtual MercerKernel &getKernel_unsafe(void)                                                svm_override { return getML().getKernel_unsafe();                            }
    virtual void prepareKernel(void)                                                            svm_override {        getML().prepareKernel(); return;                       }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        svm_override { return getML().resetKernel(modind,onlyChangeRowI,updateInfo); }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) svm_override { return getML().setKernel(xkernel,modind,onlyChangeRowI);      }

    virtual void fillCache(void) svm_override { getML().fillCache(); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) svm_override { getML().K2bypass(nv); return; }

    virtual gentype &Keqn(gentype &res,                           int resmode = 1) const svm_override { return getMLconst().Keqn(res,     resmode); }
    virtual gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const svm_override { return getMLconst().Keqn(res,altK,resmode); }
//FIXME: ADDHERE
    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getMLconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getMLconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const svm_override { return getMLconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const svm_override { return getMLconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const svm_override { return getMLconst().Km(res,xx); }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getMLconst().K2ip(res,xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const svm_override { return getMLconst().distK(xa,xb,xainf,xbinf); }
//FIXME: ADDHERE
    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getMLconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const svm_override { return getMLconst().phi2(res,ia,xa,xainf); }
//FIXME: ADDHERE
    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const svm_override { return getMLconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const svm_override { return getMLconst().phi2(res,ia,xa,xainf); }
//FIXME: ADDHERE
    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const svm_override { return getMLconst().K0ip(res,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const svm_override { return  getMLconst().K1ip(res,ia,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const svm_override { return getMLconst().K2ip(res,ia,ib,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const svm_override { return getMLconst().K3ip(res,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const svm_override { return getMLconst().K4ip(res,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const svm_override { return getMLconst().Kmip(m,res,i,pxyprod,xx,xxinfo); }
//FIXME: ADDHERE
    virtual double &K0ip(       double &res, const double &bias, const gentype **pxyprod = NULL) const svm_override { return getMLconst().K0ip(res,bias,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const svm_override { return getMLconst().K1ip(res,ia,bias,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const svm_override { return getMLconst().K2ip(res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const svm_override { return getMLconst().K3ip(res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const svm_override { return getMLconst().K4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const svm_override { return getMLconst().Kmip(m,res,i,bias,pxyprod,xx,xxinfo); }
//FIXME: ADDHERE
    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(         res,altK,pxyprod,resmode); }
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(         res     ,pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const svm_override { return getMLconst().K0(order   ,res     ,pxyprod,resmode); }
//FIXME: ADDHERE
    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const svm_override { return getMLconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }
//FIXME: ADDHERE
    virtual gentype        &K2(              gentype        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(         res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int ia, int ib, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(         res,ia,ib,altK,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual double         &K2(              double         &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(         res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(spaceDim,res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int ia, int ib                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K2(order,   res,ia,ib     ,pxyprod,xa,xb,xainfo,xbinfo,resmode); }
//FIXME: ADDHERE
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
//FIXME: ADDHERE
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const svm_override { return getMLconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
//FIXME: ADDHERE
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m         ,res,i,bias,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m         ,res,i,altK,pxyprod,xx,xxinfo,resmode); }
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m,spaceDim,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const svm_override { return getMLconst().Km(m,order   ,res,i     ,pxyprod,xx,xxinfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const svm_override { getMLconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const svm_override { getMLconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const svm_override { getMLconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual double distK(int i, int j) const svm_override { return getMLconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const svm_override { return getMLconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const svm_override { return getMLconst().denseintK(res,i,j); }

    virtual void densedKdx(double &res, int i, int j, const double &bias) const svm_override { return getMLconst().densedKdx(res,i,j,bias); }
    virtual void denseintK(double &res, int i, int j, const double &bias) const svm_override { return getMLconst().denseintK(res,i,j,bias); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const svm_override { getMLconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const svm_override { return getMLconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const svm_override { getMLconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const svm_override { getMLconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const svm_override { getMLconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm(gentype &res, int i, int j) const svm_override { return getMLconst().xelm(res,i,j); }
    virtual int xindsize(int i) const svm_override { return getMLconst().xindsize(i); }

//Variants
    int KisFullNorm       (void) const { return getKernel().isFullNorm();        }
    int KisProd           (void) const { return getKernel().isProd();            }
    int KisIndex          (void) const { return getKernel().isIndex();           }
    int KisShifted        (void) const { return getKernel().isShifted();         }
    int KisScaled         (void) const { return getKernel().isScaled();          }
    int KisShiftedScaled  (void) const { return getKernel().isShiftedScaled();   }
    int KisLeftPlain      (void) const { return getKernel().isLeftPlain();       }
    int KisRightPlain     (void) const { return getKernel().isRightPlain();      }
    int KisLeftRightPlain (void) const { return getKernel().isLeftRightPlain();  }
    int KisLeftNormal     (void) const { return getKernel().isLeftNormal();      }
    int KisRightNormal    (void) const { return getKernel().isRightNormal();     }
    int KisLeftRightNormal(void) const { return getKernel().isLeftRightNormal(); }
    int KisPartNormal     (void) const { return getKernel().isPartNormal();      }
    int KisAltDiff        (void) const { return getKernel().isAltDiff();         }
    int KneedsmProd       (void) const { return getKernel().needsmProd();        }
    int KwantsXYprod      (void) const { return getKernel().wantsXYprod();       }
    int KsuggestXYcache   (void) const { return getKernel().suggestXYcache();    }
    int KisIPdiffered     (void) const { return getKernel().isIPdiffered();      }

    int Ksize       (void) const { return getKernel().size();        }
    int KgetSymmetry(void) const { return getKernel().getSymmetry(); }

    double KcWeight(int q = 0) const { return (double) getKernel().cWeight(q); }
    int    KcType  (int q = 0) const { return          getKernel().cType(q);   }

    int KisNormalised(int q = 0) const { return getKernel().isNormalised(q); }
    int KisChained   (int q = 0) const { return getKernel().isChained(q);    }
    int KisSplit     (int q = 0) const { return getKernel().isSplit(q);      }
    int KisMulSplit  (int q = 0) const { return getKernel().isMulSplit(q);   }
    int KisMagTerm   (int q = 0) const { return getKernel().isMagTerm(q);    }

    int KnumSplits   (void) const { return getKernel().numSplits();    }
    int KnumMulSplits(void) const { return getKernel().numMulSplits(); }

    double KcRealConstants(int q = 0, int i = 0) const { return (double) getKernel().cRealConstants(q)(i); }
    int    KcIntConstants (int q = 0, int i = 0) const { return          getKernel().cIntConstants(q)(i);  }

    double KgetRealConstZero(int q = 0) const { return (double) getKernel().getRealConstZero(q); }
    int    KgetIntConstZero (int q = 0) const { return          getKernel().getIntConstZero(q);  }

    int KisKVarianceNZ(void) const { return getKernel().isKVarianceNZ(); }

    void Kadd   (int q)     { prepareKernel(); getKernel_unsafe().add(q);        resetKernel(); return; }
    void Kremove(int q)     { prepareKernel(); getKernel_unsafe().remove(q);     resetKernel(); return; }
    void Kresize(int nsize) { prepareKernel(); getKernel_unsafe().resize(nsize); resetKernel(); return; }

    void KsetFullNorm       (void) { prepareKernel(); getKernel_unsafe().setFullNorm();        resetKernel(); return; }
    void KsetNoFullNorm     (void) { prepareKernel(); getKernel_unsafe().setNoFullNorm();      resetKernel(); return; }
    void KsetProd           (void) { prepareKernel(); getKernel_unsafe().setProd();            resetKernel(); return; }
    void KsetnonProd        (void) { prepareKernel(); getKernel_unsafe().setnonProd();         resetKernel(); return; }
    void KsetLeftPlain      (void) { prepareKernel(); getKernel_unsafe().setLeftPlain();       resetKernel(); return; }
    void KsetRightPlain     (void) { prepareKernel(); getKernel_unsafe().setRightPlain();      resetKernel(); return; }
    void KsetLeftRightPlain (void) { prepareKernel(); getKernel_unsafe().setLeftRightPlain();  resetKernel(); return; }
    void KsetLeftNormal     (void) { prepareKernel(); getKernel_unsafe().setLeftNormal();      resetKernel(); return; }
    void KsetRightNormal    (void) { prepareKernel(); getKernel_unsafe().setRightNormal();     resetKernel(); return; }
    void KsetLeftRightNormal(void) { prepareKernel(); getKernel_unsafe().setLeftRightNormal(); resetKernel(); return; }

    void KsetAltDiff       (int nv) { prepareKernel(); getKernel_unsafe().setAltDiff(nv);        resetKernel(); return; }
    void KsetsuggestXYcache(int nv) { prepareKernel(); getKernel_unsafe().setsuggestXYcache(nv); resetKernel(); return; }
    void KsetIPdiffered    (int nv) { prepareKernel(); getKernel_unsafe().setIPdiffered(nv);     resetKernel(); return; }

    void KsetChained   (int q = 0) { prepareKernel(); getKernel_unsafe().setChained(q);    resetKernel(); return; }
    void KsetNormalised(int q = 0) { prepareKernel(); getKernel_unsafe().setNormalised(q); resetKernel(); return; }
    void KsetSplit     (int q = 0) { prepareKernel(); getKernel_unsafe().setSplit(q);      resetKernel(); return; }
    void KsetMulSplit  (int q = 0) { prepareKernel(); getKernel_unsafe().setMulSplit(q);   resetKernel(); return; }
    void KsetMagTerm   (int q = 0) { prepareKernel(); getKernel_unsafe().setMagTerm(q);    resetKernel(); return; }

    void KsetUnChained   (int q = 0) { prepareKernel(); getKernel_unsafe().setUnChained(q);    resetKernel(); return; }
    void KsetUnNormalised(int q = 0) { prepareKernel(); getKernel_unsafe().setUnNormalised(q); resetKernel(); return; }
    void KsetUnSplit     (int q = 0) { prepareKernel(); getKernel_unsafe().setUnSplit(q);      resetKernel(); return; }
    void KsetUnMulSplit  (int q = 0) { prepareKernel(); getKernel_unsafe().setUnMulSplit(q);   resetKernel(); return; }
    void KsetUnMagTerm   (int q = 0) { prepareKernel(); getKernel_unsafe().setUnMagTerm(q);    resetKernel(); return; }

    void KsetWeight (double nwa,  int q = 0) { gentype nw(nwa); prepareKernel(); getKernel_unsafe().setWeight(nw,q);       resetKernel(); return; }
    void KsetType   (int ndtype,  int q = 0) {                  prepareKernel(); getKernel_unsafe().setType(ndtype,q);     resetKernel(); return; }
    void KsetAltCall(int newMLid, int q = 0) {                  prepareKernel(); getKernel_unsafe().setAltCall(newMLid,q); resetKernel(); return; }

    void KsetRealConstants(double nv, int q = 0, int i = 0) { Vector<gentype> v(getKernel().cRealConstants(q)); v("&",i) = nv; prepareKernel(); getKernel_unsafe().setRealConstants(v,q); resetKernel(); return; }
    void KsetIntConstants (int nv,    int q = 0, int i = 0) { Vector<int> v(getKernel().cIntConstants(q));      v("&",i) = nv; prepareKernel(); getKernel_unsafe().setIntConstants(v,q);  resetKernel(); return; }

    void KsetRealConstZero(double nv, int q = 0) { prepareKernel(); getKernel_unsafe().setRealConstZero(nv,q); resetKernel(); return; }
    void KsetIntConstZero (int nv,    int q = 0) { prepareKernel(); getKernel_unsafe().setIntConstZero(nv,q);  resetKernel(); return; }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override { return getML().addTrainingVector (i,z,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1) svm_override { return getML().qaddTrainingVector(i,z,x,Cweigh,epsweigh); }

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getML().addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getML().addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) svm_override { return getML().addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override { return getML().addTrainingVector (i,z,x,Cweigh,epsweigh); }
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh) svm_override { return getML().qaddTrainingVector(i,z,x,Cweigh,epsweigh); }

    virtual int removeTrainingVector(int i)                                       svm_override { return getML().removeTrainingVector(i);     }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) svm_override { return getML().removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                              svm_override { return getML().removeTrainingVector(i,num); }

    virtual int setx(int                i, const SparseVector<gentype>          &x) svm_override { return getML().setx(i,x); }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) svm_override { return getML().setx(i,x); }
    virtual int setx(                      const Vector<SparseVector<gentype> > &x) svm_override { return getML().setx(x);   }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) svm_override { return getML().qswapx(i,x,dontupdate); }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) svm_override { return getML().qswapx(i,x,dontupdate); }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) svm_override { return getML().qswapx(  x,dontupdate); }

    virtual int sety(int                i, const gentype         &y) svm_override { return getML().sety(i,y); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &y) svm_override { return getML().sety(i,y); }
    virtual int sety(                      const Vector<gentype> &y) svm_override { return getML().sety(y);   }

    virtual int sety(int                i, double                z) svm_override { return getML().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) svm_override { return getML().sety(i,z); }
    virtual int sety(                      const Vector<double> &z) svm_override { return getML().sety(z); }

    virtual int sety(int                i, const Vector<double>          &z) svm_override { return getML().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) svm_override { return getML().sety(i,z); }
    virtual int sety(                      const Vector<Vector<double> > &z) svm_override { return getML().sety(z); }

    virtual int sety(int                i, const d_anion         &z) svm_override { return getML().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) svm_override { return getML().sety(i,z); }
    virtual int sety(                      const Vector<d_anion> &z) svm_override { return getML().sety(z); }

    virtual int setd(int                i, int                d) svm_override { return getML().setd(i,d); }
    virtual int setd(const Vector<int> &i, const Vector<int> &d) svm_override { return getML().setd(i,d); }
    virtual int setd(                      const Vector<int> &d) svm_override { return getML().setd(d);   }

    virtual int setCweight(int i,                double nv               ) svm_override { return getML().setCweight(i,nv); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getML().setCweight(i,nv); }
    virtual int setCweight(                      const Vector<double> &nv) svm_override { return getML().setCweight(nv);   }

    virtual int setCweightfuzz(int i,                double nv               ) svm_override { return getML().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &nv) svm_override { return getML().setCweightfuzz(i,nv); }
    virtual int setCweightfuzz(                      const Vector<double> &nv) svm_override { return getML().setCweightfuzz(nv);   }

    virtual int setsigmaweight(int i,                double nv               ) svm_override { return getML().setsigmaweight(i,nv); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getML().setsigmaweight(i,nv); }
    virtual int setsigmaweight(                      const Vector<double> &nv) svm_override { return getML().setsigmaweight(  nv); }

    virtual int setepsweight(int i,                double nv               ) svm_override { return getML().setepsweight(i,nv); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &nv) svm_override { return getML().setepsweight(i,nv); }
    virtual int setepsweight(                      const Vector<double> &nv) svm_override { return getML().setepsweight(nv);   }

    virtual int scaleCweight    (double s) svm_override { return getML().scaleCweight(s);     }
    virtual int scaleCweightfuzz(double s) svm_override { return getML().scaleCweightfuzz(s); }
    virtual int scalesigmaweight(double s) svm_override { return getML().scalesigmaweight(s); }
    virtual int scaleepsweight  (double s) svm_override { return getML().scaleepsweight(s);   }

    virtual void assumeConsistentX  (void) svm_override { getML().assumeConsistentX  (); return; }
    virtual void assumeInconsistentX(void) svm_override { getML().assumeInconsistentX(); return; }

    virtual int isXConsistent(void)        const svm_override { return getMLconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const svm_override { return getMLconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) svm_override { getML().xferx(xsrc); return; }

    virtual const vecInfo &xinfo(int i)                       const svm_override { return getMLconst().xinfo(i); }
    virtual int xtang(int i)                                  const svm_override { return getMLconst().xtang(i); }
    virtual const SparseVector<gentype> &x(int i)             const svm_override { return getMLconst().x(i); }
    virtual int xisrank(int i)                                const svm_override { return getMLconst().xisrank(i);  }
    virtual int xisgrad(int i)                                const svm_override { return getMLconst().xisgrad(i);  }
    virtual int xisrankorgrad(int i)                          const svm_override { return getMLconst().xisrankorgrad(i);  }
    virtual int xisclass(int i, int defaultclass, int q = -1) const svm_override { return getMLconst().xisclass(i,defaultclass,q); }
    virtual const gentype &y(int i)                           const svm_override { return getMLconst().y(i); }

    // Basis stuff

    virtual int NbasisUU(void)    const svm_override { return getMLconst().NbasisUU();    }
    virtual int basisTypeUU(void) const svm_override { return getMLconst().basisTypeUU(); }
    virtual int defProjUU(void)   const svm_override { return getMLconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const svm_override { return getMLconst().VbasisUU(); }

    virtual int setBasisYUU(void)                     svm_override { return getML().setBasisYUU();             }
    virtual int setBasisUUU(void)                     svm_override { return getML().setBasisUUU();             }
    virtual int addToBasisUU(int i, const gentype &o) svm_override { return getML().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU(int i)              svm_override { return getML().removeFromBasisUU(i);      }
    virtual int setBasisUU(int i, const gentype &o)   svm_override { return getML().setBasisUU(i,o);           }
    virtual int setBasisUU(const Vector<gentype> &o)  svm_override { return getML().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)         svm_override { return getML().setDefaultProjectionUU(d); }
    virtual int setBasisUU(int n, int d)              svm_override { return getML().setBasisUU(n,d);           }

    virtual int NbasisVV(void)    const svm_override { return getMLconst().NbasisVV();    }
    virtual int basisTypeVV(void) const svm_override { return getMLconst().basisTypeVV(); }
    virtual int defProjVV(void)   const svm_override { return getMLconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const svm_override { return getMLconst().VbasisVV(); }

    virtual int setBasisYVV(void)                     svm_override { return getML().setBasisYVV();             }
    virtual int setBasisUVV(void)                     svm_override { return getML().setBasisUVV();             }
    virtual int addToBasisVV(int i, const gentype &o) svm_override { return getML().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV(int i)              svm_override { return getML().removeFromBasisVV(i);      }
    virtual int setBasisVV(int i, const gentype &o)   svm_override { return getML().setBasisVV(i,o);           }
    virtual int setBasisVV(const Vector<gentype> &o)  svm_override { return getML().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)         svm_override { return getML().setDefaultProjectionVV(d); }
    virtual int setBasisVV(int n, int d)              svm_override { return getML().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel(void) const                  svm_override { return getMLconst().getUUOutputKernel();          }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       svm_override { return getML().getUUOutputKernel_unsafe();        }
    virtual int resetUUOutputKernel(int modind = 1)                            svm_override { return getML().resetUUOutputKernel(modind);       }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) svm_override { return getML().setUUOutputKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) svm_override { return getML().randomise(sparsity); }
    virtual int autoen(void)               svm_override { return getML().autoen();            }
    virtual int renormalise(void)          svm_override { return getML().renormalise();       }
    virtual int realign(void)              svm_override { return getML().realign();           }

    virtual int setzerotol(double zt)                 svm_override { return getML().setzerotol(zt);                 }
    virtual int setOpttol(double xopttol)             svm_override { return getML().setOpttol(xopttol);             }
    virtual int setmaxitcnt(int xmaxitcnt)            svm_override { return getML().setmaxitcnt(xmaxitcnt);         }
    virtual int setmaxtraintime(double xmaxtraintime) svm_override { return getML().setmaxtraintime(xmaxtraintime); }

    virtual int setmaxitermvrank(int nv) svm_override { return getML().setmaxitermvrank(nv); }
    virtual int setlrmvrank(double nv)   svm_override { return getML().setlrmvrank(nv);      }
    virtual int setztmvrank(double nv)   svm_override { return getML().setztmvrank(nv);      }

    virtual int setbetarank(double nv) svm_override { return getML().setbetarank(nv); }

    virtual int setC    (double xC)             svm_override { return getML().setC(xC);            }
    virtual int setsigma(double xsigma)         svm_override { return getML().setsigma(xsigma);    }
    virtual int seteps  (double xeps)           svm_override { return getML().seteps(xeps);        }
    virtual int setCclass  (int d, double xC)   svm_override { return getML().setCclass(d,xC);     }
    virtual int setepsclass(int d, double xeps) svm_override { return getML().setepsclass(d,xeps); }

    virtual int scale(double a) svm_override { return getML().scale(a);  }
    virtual int reset(void)     svm_override { return getML().reset();   }
    virtual int restart(void)   svm_override { return getML().restart(); }
    virtual int home(void)      svm_override { return getML().home();    }

    virtual ML_Base &operator*=(double sf) svm_override { scale(sf); return *this; }

    virtual int scaleby(double sf) svm_override { *this *= sf; return 1; }

    virtual int settspaceDim(int newdim) svm_override { return getML().settspaceDim(newdim); }
    virtual int addtspaceFeat(int i)     svm_override { return getML().addtspaceFeat(i);     }
    virtual int removetspaceFeat(int i)  svm_override { return getML().removetspaceFeat(i);  }
    virtual int addxspaceFeat(int i)     svm_override { return getML().addxspaceFeat(i);     }
    virtual int removexspaceFeat(int i)  svm_override { return getML().removexspaceFeat(i);  }

    virtual int setsubtype(int i) svm_override { return getML().setsubtype(i); }

    virtual int setorder(int neword)                 svm_override { return getML().setorder(neword);        }
    virtual int addclass(int label, int epszero = 0) svm_override { return getML().addclass(label,epszero); }

    // Sampling mode

    virtual int isSampleMode(void) const svm_override { return getMLconst().isSampleMode(); }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1, int sampType = 0) svm_override { return getML().setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType); }

    // Training functions:

    virtual void fudgeOn(void)  svm_override { getML().fudgeOn();  return; }
    virtual void fudgeOff(void) svm_override { getML().fudgeOff(); return; }

#ifndef SWIG
    virtual int train(int &res) svm_override { svmvolatile int killSwitch = 0; return train(res,killSwitch); }
    virtual int train(int &res, svmvolatile int &killSwitch) svm_override { return getML().train(res,killSwitch); }
#endif

//Variants
    virtual int train(void) { int res = 0; svmvolatile int killSwitch = 0; train(res,killSwitch); return res; }

    // Evaluation Functions:
#ifndef SWIG
    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getMLconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const svm_override { return getMLconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getMLconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual double eTrainingVector(int i) const svm_override { return getMLconst().eTrainingVector(i); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const svm_override { return getMLconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double         &dedgTrainingVector(double         &res, int i) const svm_override { return getMLconst().dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const svm_override { return getMLconst().dedgTrainingVector(res,i); }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const svm_override { return getMLconst().dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const svm_override { return getMLconst().dedgTrainingVector(res,i); }

    virtual double dedgAlphaTrainingVector(int i, int j) const svm_override { return getMLconst().dedgAlphaTrainingVector(i,j); }
    virtual Matrix<double> &dedgAlphaTrainingVector(Matrix<double> &res) const svm_override { return getMLconst().dedgAlphaTrainingVector(res); }
//FIXME: ADDHERE
    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const svm_override { getMLconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const svm_override { getMLconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, int i) const svm_override { getMLconst().deTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const svm_override { getMLconst().deTrainingVectorX(resx,i); return; }

    virtual int ggTrainingVector(double         &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getMLconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getMLconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion        &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const svm_override { return getMLconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
//FIXME: ADDHERE
    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const svm_override { getMLconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const svm_override { getMLconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const svm_override { getMLconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const svm_override { getMLconst().dgTrainingVector(res,resn,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, gentype &resn, int i) const svm_override { getMLconst().deTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const svm_override { getMLconst().dgTrainingVector(res,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const svm_override { getMLconst().deTrainingVector(res,i); return; }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const svm_override { return getMLconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x,                  const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().gg(     resg,x        ,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x,                  const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().hh(resh,     x        ,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double e(const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { return getMLconst().e(y,x,xinf); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const svm_override { return getMLconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }
//FIXME: ADDHERE
    virtual void dedg(double         &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(Vector<double> &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(d_anion        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(gentype        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dedg(res,y,x,xinf); return; }

    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dgX(resx,x,xinf); return; }
    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dgX(resx,x,xinf); return; }

    virtual void deX(Vector<gentype> &resx, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().deX(resx,y,x,xinf); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const svm_override { return getMLconst().gg(resg,x,retaltg,xinf,pxyprodx); }
//FIXME: ADDHERE
    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().dg(res,resn,x,xinf); return; }

    virtual void de(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const svm_override { getMLconst().de(res,resn,y,x,xinf); return; }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const svm_override { getMLconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }
#endif
//Variants
    virtual double ggTrainingVector(int i) const { gentype res; ggTrainingVector(res,i); return (double) res; }
    virtual double hhTrainingVector(int i) const { gentype res; hhTrainingVector(res,i); return (double) res; }
    virtual double covTrainingVector(int i, int j) const { gentype res; gentype dummy; covTrainingVector(res,dummy,i,j); return (double) res; }

//Variants
    virtual double gg(double *xxa, int dima) const
    {
        gentype res;
        SparseVector<gentype> x;
        int i;

        for ( i = 0 ; i < dima ; i++ )
        {
            x("&",i) = xxa[i];
        }

        gg(res,x);

        return (double) res;
    }

//Variants
    virtual double hh(double *xxa, int dima) const
    {
        gentype res;
        SparseVector<gentype> x;
        int i;

        for ( i = 0 ; i < dima ; i++ )
        {
            x("&",i) = xxa[i];
        }

        hh(res,x);

        return (double) res;
    }

//Variants
    virtual double cov(double *xxa, int dima, double *xxb, int dimb) const
    {
        gentype res;
        gentype dummy;
        SparseVector<gentype> xa;
        SparseVector<gentype> xb;
        int i;

        for ( i = 0 ; i < dima ; i++ )
        {
            xa("&",i) = xxa[i];
        }

        for ( i = 0 ; i < dimb ; i++ )
        {
            xb("&",i) = xxb[i];
        }

        cov(res,dummy,xa,xb);

        return (double) res;
    }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const svm_override { return getMLconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const svm_override { return getMLconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const svm_override { return getMLconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const svm_override { return getMLconst().covar(resv,x); }

//Variants
    virtual double varTrainingVector(int i) const { gentype res; gentype dummy; varTrainingVector(res,dummy,i); return (double) res; }

//Variants
    virtual double var(double *xxa, int dima) const
    {
        gentype res;
        gentype dummy;
        SparseVector<gentype> xa;
        int i;

        for ( i = 0 ; i < dima ; i++ )
        {
            xa("&",i) = xxa[i];
        }

        var(res,dummy,xa);

        return (double) res;
    }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const svm_override { return getMLconst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const svm_override { return getMLconst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const svm_override { return getMLconst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const svm_override { return getMLconst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) svm_override { getML().setaltx(_altxsrc); return; }

    virtual int disable(int i)                svm_override { return getML().disable(i); }
    virtual int disable(const Vector<int> &i) svm_override { return getML().disable(i); }

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum      (SparseVector<gentype> &res) const svm_override { return getMLconst().xsum(res);       }
    virtual const SparseVector<gentype> &xmean     (SparseVector<gentype> &res) const svm_override { return getMLconst().xmean(res);      }
    virtual const SparseVector<gentype> &xmeansq   (SparseVector<gentype> &res) const svm_override { return getMLconst().xmeansq(res);    }
    virtual const SparseVector<gentype> &xsqsum    (SparseVector<gentype> &res) const svm_override { return getMLconst().xsqsum(res);     }
    virtual const SparseVector<gentype> &xsqmean   (SparseVector<gentype> &res) const svm_override { return getMLconst().xsqmean(res);    }
    virtual const SparseVector<gentype> &xmedian   (SparseVector<gentype> &res) const svm_override { return getMLconst().xmedian(res);    }
    virtual const SparseVector<gentype> &xvar      (SparseVector<gentype> &res) const svm_override { return getMLconst().xvar(res);       }
    virtual const SparseVector<gentype> &xstddev   (SparseVector<gentype> &res) const svm_override { return getMLconst().xstddev(res);    }
    virtual const SparseVector<gentype> &xmax      (SparseVector<gentype> &res) const svm_override { return getMLconst().xmax(res);       }
    virtual const SparseVector<gentype> &xmin      (SparseVector<gentype> &res) const svm_override { return getMLconst().xmin(res);       }

    // Kernel normalisation function

    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) svm_override { return getML().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) svm_override { return getML().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) svm_override { return getML().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const svm_override { return getMLconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const svm_override { return getMLconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const svm_override { return getMLconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const svm_override { return getMLconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const svm_override { return getMLconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const svm_override { return getMLconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const svm_override { return getMLconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const svm_override { return getMLconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const svm_override { return getMLconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const svm_override { return getMLconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const svm_override { return getMLconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const svm_override { return getMLconst().detangle_x(i,usextang); }

    virtual int detangle_x(SparseVector<gentype> *&xuntang, vecInfo *&xinfountang,
                   const SparseVector<gentype> *&xnear, const SparseVector<gentype> *&xfar, const SparseVector<gentype> *&xfarfar, 
                   const vecInfo *&xnearinfo, const vecInfo *&xfarinfo, 
                   int &inear, int &ifar, const gentype *&ineartup, const gentype *&ifartup,
                   int &ilr, int &irr, int &igr, 
                   int &iokr, int &iok,
                   int i, int &idiagr, const SparseVector<gentype> *xx, const vecInfo *xxinfo, int &gradOrder, 
                   int &iplanr, int &iplan, int &iset, int usextang = 1, int allocxuntangifneeded = 1) const svm_override
    {
        return getMLconst().detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,gradOrder,iplanr,iplan,iset,usextang,allocxuntangifneeded);
    }










    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       { NiceAssert( ( type() >=   0 ) && ( type() <=  99 ) ); return (dynamic_cast<      SVM_Generic &>(getML     ().getML     ())).getSVM();      }
    virtual const SVM_Generic &getSVMconst(void) const { NiceAssert( ( type() >=   0 ) && ( type() <=  99 ) ); return (dynamic_cast<const SVM_Generic &>(getMLconst().getMLconst())).getSVMconst(); }

    // Constructors, destructors, assignment etc..

    virtual int setAlpha(const Vector<gentype> &newAlpha) { return getSVM().setAlpha(newAlpha); }
    virtual int setBias (const gentype         &newBias ) { return getSVM().setBias (newBias ); }

//FIXME: ADDHERE
    virtual int setAlphaR(const Vector<double>          &newAlpha) { return getSVM().setAlphaR(newAlpha); }
    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) { return getSVM().setAlphaV(newAlpha); }
    virtual int setAlphaA(const Vector<d_anion>         &newAlpha) { return getSVM().setAlphaA(newAlpha); }

//FIXME: ADDHERE
    virtual int setBiasR(const double         &newBias) { return getSVM().setBiasR(newBias); }
    virtual int setBiasV(const Vector<double> &newBias) { return getSVM().setBiasV(newBias); }
    virtual int setBiasA(const d_anion        &newBias) { return getSVM().setBiasA(newBias); }

    // Information functions (training data):

    virtual int NZ (void)  const { return getSVMconst().NZ ();  }
    virtual int NF (void)  const { return getSVMconst().NF ();  }
    virtual int NS (void)  const { return getSVMconst().NS ();  }
    virtual int NC (void)  const { return getSVMconst().NC ();  }
    virtual int NLB(void)  const { return getSVMconst().NLB();  }
    virtual int NLF(void)  const { return getSVMconst().NLF();  }
    virtual int NUF(void)  const { return getSVMconst().NUF();  }
    virtual int NUB(void)  const { return getSVMconst().NUB();  }
    virtual int NF (int q) const { return getSVMconst().NF (q); }
    virtual int NZ (int q) const { return getSVMconst().NZ (q); }
    virtual int NS (int q) const { return getSVMconst().NS (q); }
    virtual int NC (int q) const { return getSVMconst().NC (q); }
    virtual int NLB(int q) const { return getSVMconst().NLB(q); }
    virtual int NLF(int q) const { return getSVMconst().NLF(q); }
    virtual int NUF(int q) const { return getSVMconst().NUF(q); }
    virtual int NUB(int q) const { return getSVMconst().NUB(q); }

    virtual const Vector<Vector<int> > &ClassRep(void)  const { return getSVMconst().ClassRep();  }
    virtual int                         findID(int ref) const { return getSVMconst().findID(ref); }

    virtual int isLinearCost(void)    const { return getSVMconst().isLinearCost();    }
    virtual int isQuadraticCost(void) const { return getSVMconst().isQuadraticCost(); }
    virtual int is1NormCost(void)     const { return getSVMconst().is1NormCost();     }
    virtual int isVarBias(void)       const { return getSVMconst().isVarBias();       }
    virtual int isPosBias(void)       const { return getSVMconst().isPosBias();       }
    virtual int isNegBias(void)       const { return getSVMconst().isNegBias();       }
    virtual int isFixedBias(void)     const { return getSVMconst().isFixedBias();     }
    virtual int isVarBias(int q)      const { return getSVMconst().isVarBias(q);      }
    virtual int isPosBias(int q)      const { return getSVMconst().isPosBias(q);      }
    virtual int isNegBias(int q)      const { return getSVMconst().isNegBias(q);      }
    virtual int isFixedBias(int q)    const { return getSVMconst().isFixedBias(q);    } 

    virtual int isNoMonotonicConstraints(void)    const { return getSVMconst().isNoMonotonicConstraints();    }
    virtual int isForcedMonotonicIncreasing(void) const { return getSVMconst().isForcedMonotonicIncreasing(); }
    virtual int isForcedMonotonicDecreasing(void) const { return getSVMconst().isForcedMonotonicDecreasing(); }

    virtual int isOptActive(void) const { return getSVMconst().isOptActive(); }
    virtual int isOptSMO(void)    const { return getSVMconst().isOptSMO();    }
    virtual int isOptD2C(void)    const { return getSVMconst().isOptD2C();    }
    virtual int isOptGrad(void)   const { return getSVMconst().isOptGrad();   }

    virtual int isFixedTube(void)  const { return getSVMconst().isFixedTube();  }
    virtual int isShrinkTube(void) const { return getSVMconst().isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const { return getSVMconst().isRestrictEpsPos(); }
    virtual int isRestrictEpsNeg(void) const { return getSVMconst().isRestrictEpsNeg(); }

    virtual int isClassifyViaSVR(void) const { return getSVMconst().isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const { return getSVMconst().isClassifyViaSVM(); }

    virtual int is1vsA(void)    const { return getSVMconst().is1vsA();    }
    virtual int is1vs1(void)    const { return getSVMconst().is1vs1();    }
    virtual int isDAGSVM(void)  const { return getSVMconst().isDAGSVM();  }
    virtual int isMOC(void)     const { return getSVMconst().isMOC();     }
    virtual int ismaxwins(void) const { return getSVMconst().ismaxwins(); }
    virtual int isrecdiv(void)  const { return getSVMconst().isrecdiv();  }

    virtual int isatonce(void) const { return getSVMconst().isatonce(); }
    virtual int isredbin(void) const { return getSVMconst().isredbin(); }

    virtual int isKreal(void)   const { return getSVMconst().isKreal();   }
    virtual int isKunreal(void) const { return getSVMconst().isKunreal(); }

    virtual int isanomalyOn(void)  const { return getSVMconst().isanomalyOn();  }
    virtual int isanomalyOff(void) const { return getSVMconst().isanomalyOff(); }

    virtual int isautosetOff(void)          const { return getSVMconst().isautosetOff();          }
    virtual int isautosetCscaled(void)      const { return getSVMconst().isautosetCscaled();      }
    virtual int isautosetCKmean(void)       const { return getSVMconst().isautosetCKmean();       }
    virtual int isautosetCKmedian(void)     const { return getSVMconst().isautosetCKmedian();     }
    virtual int isautosetCNKmean(void)      const { return getSVMconst().isautosetCNKmean();      }
    virtual int isautosetCNKmedian(void)    const { return getSVMconst().isautosetCNKmedian();    }
    virtual int isautosetLinBiasForce(void) const { return getSVMconst().isautosetLinBiasForce(); }

    virtual double outerlr(void)       const { return getSVMconst().outerlr();       }
    virtual double outermom(void)      const { return getSVMconst().outermom();      }
    virtual int    outermethod(void)   const { return getSVMconst().outermethod();   }
    virtual double outertol(void)      const { return getSVMconst().outertol();      }
    virtual double outerovsc(void)     const { return getSVMconst().outerovsc();     }
    virtual int    outermaxitcnt(void) const { return getSVMconst().outermaxitcnt(); }
    virtual int    outermaxcache(void) const { return getSVMconst().outermaxcache(); }

    virtual       int      maxiterfuzzt(void) const { return getSVMconst().maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return getSVMconst().usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return getSVMconst().lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return getSVMconst().ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return getSVMconst().costfnfuzzt();  }

    virtual int m(void) const { return getSVMconst().m(); }

    virtual double LinBiasForce(void)   const { return getSVMconst().LinBiasForce();   }
    virtual double QuadBiasForce(void)  const { return getSVMconst().QuadBiasForce();  }
    virtual double LinBiasForce(int q)  const { return getSVMconst().LinBiasForce(q);  }
    virtual double QuadBiasForce(int q) const { return getSVMconst().QuadBiasForce(q); }

    virtual double nu(void)     const { return getSVMconst().nu();     }
    virtual double nuQuad(void) const { return getSVMconst().nuQuad(); }

    virtual double theta(void)   const { return getSVMconst().theta();   }
    virtual int    simnorm(void) const { return getSVMconst().simnorm(); }

    virtual double anomalyNu(void)    const { return getSVMconst().anomalyNu();    }
    virtual int    anomalyClass(void) const { return getSVMconst().anomalyClass(); }

    virtual double autosetCval(void)  const { return getSVMconst().autosetCval();  }
    virtual double autosetnuval(void) const { return getSVMconst().autosetnuval(); }

    virtual int anomclass(void)          const { return getSVMconst().anomclass();       }
    virtual int singmethod(void)         const { return getSVMconst().singmethod();      }
    virtual double rejectThreshold(void) const { return getSVMconst().rejectThreshold(); }

//FIXME: ADDHERE
    virtual const Matrix<double>          &Gp         (void)        const { return getSVMconst().Gp();        }
    virtual const Matrix<double>          &XX         (void)        const { return getSVMconst().XX();        }
    virtual const Vector<double>          &kerndiag   (void)        const { return getSVMconst().kerndiag();  }
    virtual const Vector<Vector<double> > &getu       (void)        const { return getSVMconst().getu();      }
    virtual const gentype                 &bias       (void)        const { return getSVMconst().bias();      }
    virtual const Vector<gentype>         &alpha      (void)        const { return getSVMconst().alpha();     }
    virtual const Vector<double>          &zR         (void)        const { return getSVMconst().zR();        }
    virtual const Vector<Vector<double> > &zV         (void)        const { return getSVMconst().zV();        }
    virtual const Vector<d_anion>         &zA         (void)        const { return getSVMconst().zA();        }
    virtual const double                  &biasR      (void)        const { return getSVMconst().biasR();     }
    virtual const Vector<double>          &biasV      (int raw = 0) const { return getSVMconst().biasV(raw);  }
    virtual const d_anion                 &biasA      (void)        const { return getSVMconst().biasA();     }
    virtual const Vector<double>          &alphaR     (void)        const { return getSVMconst().alphaR();    }
    virtual const Vector<Vector<double> > &alphaV     (int raw = 0) const { return getSVMconst().alphaV(raw); }
    virtual const Vector<d_anion>         &alphaA     (void)        const { return getSVMconst().alphaA();    }

    virtual const double         &zR(int i) const { return getSVMconst().zR(i); }
    virtual const Vector<double> &zV(int i) const { return getSVMconst().zV(i); }
    virtual const d_anion        &zA(int i) const { return getSVMconst().zA(i); }
//Variants
    virtual void npkerndiag(double **res, int *dim) const { *dim = kerndiag().size(); *res = const_cast<double *>(&(kerndiag()(zeroint()))); return; }
    virtual void npzR      (double **res, int *dim) const { *dim = zR().size();       *res = const_cast<double *>(&(zR()(zeroint())));       return; }
    virtual void npbiasV   (double **res, int *dim) const { *dim = biasV().size();    *res = const_cast<double *>(&(biasV()(zeroint())));    return; }
    virtual void npalphaR  (double **res, int *dim) const { *dim = alphaR().size();   *res = const_cast<double *>(&(alphaR()(zeroint())));   return; }

    // Training set modification:

    virtual int removeNonSupports(void)      { return getSVM().removeNonSupports();      }
    virtual int trimTrainingSet(int maxsize) { return getSVM().trimTrainingSet(maxsize); }

    // General modification and autoset functions

    virtual int setLinearCost(void)                        { return getSVM().setLinearCost();         }
    virtual int setQuadraticCost(void)                     { return getSVM().setQuadraticCost();      }
    virtual int set1NormCost(void)                         { return getSVM().set1NormCost();          }
    virtual int setVarBias(void)                           { return getSVM().setVarBias();            }
    virtual int setPosBias(void)                           { return getSVM().setPosBias();            }
    virtual int setNegBias(void)                           { return getSVM().setNegBias();            }
    virtual int setFixedBias(double newbias = 0.0)         { return getSVM().setFixedBias(newbias);   }
    virtual int setVarBias(int q)                          { return getSVM().setVarBias(q);           }
    virtual int setPosBias(int q)                          { return getSVM().setPosBias(q);           }
    virtual int setNegBias(int q)                          { return getSVM().setNegBias(q);           }
    virtual int setFixedBias(int q, double newbias = 0.0)  { return getSVM().setFixedBias(q,newbias); }
    virtual int setFixedBias(const gentype &newbias)       { return getSVM().setFixedBias(newbias);   }

    virtual int setNoMonotonicConstraints(void)    { return getSVM().setNoMonotonicConstraints();    }
    virtual int setForcedMonotonicIncreasing(void) { return getSVM().setForcedMonotonicIncreasing(); }
    virtual int setForcedMonotonicDecreasing(void) { return getSVM().setForcedMonotonicDecreasing(); }

    virtual int setOptActive(void) { return getSVM().setOptActive(); }
    virtual int setOptSMO(void)    { return getSVM().setOptSMO();    }
    virtual int setOptD2C(void)    { return getSVM().setOptD2C();    }
    virtual int setOptGrad(void)   { return getSVM().setOptGrad();   }

    virtual int setFixedTube(void)  { return getSVM().setFixedTube();  }
    virtual int setShrinkTube(void) { return getSVM().setShrinkTube(); }

    virtual int setRestrictEpsPos(void) { return getSVM().setRestrictEpsPos(); }
    virtual int setRestrictEpsNeg(void) { return getSVM().setRestrictEpsNeg(); }

    virtual int setClassifyViaSVR(void) { return getSVM().setClassifyViaSVR(); }
    virtual int setClassifyViaSVM(void) { return getSVM().setClassifyViaSVM(); }

    virtual int set1vsA(void)    { return getSVM().set1vsA();    }
    virtual int set1vs1(void)    { return getSVM().set1vs1();    }
    virtual int setDAGSVM(void)  { return getSVM().setDAGSVM();  }
    virtual int setMOC(void)     { return getSVM().setMOC();     }
    virtual int setmaxwins(void) { return getSVM().setmaxwins(); }
    virtual int setrecdiv(void)  { return getSVM().setrecdiv();  }

    virtual int setatonce(void) { return getSVM().setatonce(); }
    virtual int setredbin(void) { return getSVM().setredbin(); }

    virtual int setKreal(void)   { return getSVM().setKreal();   }
    virtual int setKunreal(void) { return getSVM().setKunreal(); }

    virtual int anomalyOn(int danomalyClass, double danomalyNu) { return getSVM().anomalyOn(danomalyClass,danomalyNu); }
    virtual int anomalyOff(void)                                { return getSVM().anomalyOff();                        }

    virtual int setouterlr(double xouterlr)           { return getSVM().setouterlr(xouterlr);              }
    virtual int setoutermom(double xoutermom)         { return getSVM().setoutermom(xoutermom);            }
    virtual int setoutermethod(int xoutermethod)      { return getSVM().setoutermethod(xoutermethod);      }
    virtual int setoutertol(double xoutertol)         { return getSVM().setoutertol(xoutertol);            }
    virtual int setouterovsc(double xouterovsc)       { return getSVM().setouterovsc(xouterovsc);          }
    virtual int setoutermaxitcnt(int xoutermaxits)    { return getSVM().setoutermaxitcnt(xoutermaxits);    }
    virtual int setoutermaxcache(int xoutermaxcacheN) { return getSVM().setoutermaxcache(xoutermaxcacheN); }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)              { return getSVM().setmaxiterfuzzt(xmaxiterfuzzt); }
    virtual int setusefuzzt(int xusefuzzt)                      { return getSVM().setusefuzzt(xusefuzzt);         }
    virtual int setlrfuzzt(double xlrfuzzt)                     { return getSVM().setlrfuzzt(xlrfuzzt);           }
    virtual int setztfuzzt(double xztfuzzt)                     { return getSVM().setztfuzzt(xztfuzzt);           }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)     { return getSVM().setcostfnfuzzt(xcostfnfuzzt);   }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt) { return getSVM().setcostfnfuzzt(xcostfnfuzzt);   }

    virtual int setm(int xm) { return getSVM().setm(xm); }

    virtual int setLinBiasForce(double newval)         { return getSVM().setLinBiasForce(newval);    }
    virtual int setQuadBiasForce(double newval)        { return getSVM().setQuadBiasForce(newval);   }
    virtual int setLinBiasForce(int q, double newval)  { return getSVM().setLinBiasForce(q,newval);  }
    virtual int setQuadBiasForce(int q, double newval) { return getSVM().setQuadBiasForce(q,newval); }

    virtual int setnu(double xnu)         { return getSVM().setnu(xnu);         }
    virtual int setnuQuad(double xnuQuad) { return getSVM().setnuQuad(xnuQuad); }

    virtual int settheta(double nv) { return getSVM().settheta(nv);   }
    virtual int setsimnorm(int nv)  { return getSVM().setsimnorm(nv); }

    virtual int autosetOff(void)                                     { return getSVM().autosetOff();                    }
    virtual int autosetCscaled(double Cval)                          { return getSVM().autosetCscaled(Cval);            }
    virtual int autosetCKmean(void)                                  { return getSVM().autosetCKmean();                 }
    virtual int autosetCKmedian(void)                                { return getSVM().autosetCKmedian();               }
    virtual int autosetCNKmean(void)                                 { return getSVM().autosetCNKmean();                }
    virtual int autosetCNKmedian(void)                               { return getSVM().autosetCNKmedian();              }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { return getSVM().autosetLinBiasForce(nuval,Cval); }

    virtual void setanomalyclass(int n)        { getSVM().setanomalyclass(n);     return; }
    virtual void setsingmethod(int nv)         { getSVM().setsingmethod(nv);      return; }
    virtual void setRejectThreshold(double nv) { getSVM().setRejectThreshold(nv); return; }

    // Likelihood

    virtual double quasiloglikelihood(void) const { return getSVMconst().quasiloglikelihood(); }





    // ================================================================
    //     Common functions for all ONNs
    // ================================================================

#ifndef SWIG
    virtual       ONN_Generic &getONN     (void)       { NiceAssert( ( type() >= 100 ) && ( type() <= 199 ) ); return (dynamic_cast<      ONN_Generic &>(getML     ().getML     ())).getONN();      }
    virtual const ONN_Generic &getONNconst(void) const { NiceAssert( ( type() >= 100 ) && ( type() <= 199 ) ); return (dynamic_cast<const ONN_Generic &>(getMLconst().getMLconst())).getONNconst(); }
#endif

    // Information functions (training data):

    virtual double lr(void) const { return getONNconst().lr(); }

    virtual const SparseVector<gentype> &W(void) const { return getONNconst().W(); }
    virtual const gentype               &B(void) const { return getONNconst().B(); }

    // General modification and autoset functions

    virtual int setlr(double xlr) { return getONN().setlr(xlr); }

    virtual int setW(const SparseVector<gentype> &Wsrc) { return getONN().setW(Wsrc); }
    virtual int setB(const gentype               &bsrc) { return getONN().setB(bsrc); }









    // ================================================================
    //     Common functions for all BLKs
    // ================================================================

#ifndef SWIG
    virtual       BLK_Generic &getBLK     (void)       { NiceAssert( ( type() >= 200 ) && ( type() <= 299 ) ); return (dynamic_cast<      BLK_Generic &>(getML     ().getML     ())).getBLK();      }
    virtual const BLK_Generic &getBLKconst(void) const { NiceAssert( ( type() >= 200 ) && ( type() <= 299 ) ); return (dynamic_cast<const BLK_Generic &>(getMLconst().getMLconst())).getBLKconst(); }
#endif

    // Information functions (training data):

    virtual const gentype &outfn    (void) const { return getBLKconst().outfn();     }
    virtual const gentype &outfngrad(void) const { return getBLKconst().outfngrad(); }

    // General modification and autoset functions

    virtual int setoutfn(const gentype &newoutfn)     { return getBLK().setoutfn(newoutfn); }
    virtual int setoutfn(const std::string &newoutfn) { return getBLK().setoutfn(newoutfn); }

    // Streams used by userio

    virtual int setuseristream(std::istream &src) { return getBLK().setuseristream(src); }
    virtual int setuserostream(std::ostream &dst) { return getBLK().setuserostream(dst); }

    virtual std::istream &useristream(void) const { return getBLKconst().useristream(); }
    virtual std::ostream &userostream(void) const { return getBLKconst().userostream(); }

    // Callback function used by calbak

    virtual int setcallback(int (*ncallback)(gentype &, const SparseVector<gentype> &, void *), void *ncallbackfndata) { return getBLK().setcallback(ncallback,ncallbackfndata); }

    virtual int setK0callback(K0callbackfn nK0callback, void *nK0callbackdata) { return getBLK().setK0callback(nK0callback,nK0callbackdata); }
    virtual int setK1callback(K1callbackfn nK1callback, void *nK1callbackdata) { return getBLK().setK1callback(nK1callback,nK1callbackdata); }
    virtual int setK2callback(K2callbackfn nK2callback, void *nK2callbackdata) { return getBLK().setK2callback(nK2callback,nK2callbackdata); }
    virtual int setK3callback(K3callbackfn nK3callback, void *nK3callbackdata) { return getBLK().setK3callback(nK3callback,nK3callbackdata); }
    virtual int setK4callback(K4callbackfn nK4callback, void *nK4callbackdata) { return getBLK().setK4callback(nK4callback,nK4callbackdata); }
    virtual int setKmcallback(Kmcallbackfn nKmcallback, void *nKmcallbackdata) { return getBLK().setKmcallback(nKmcallback,nKmcallbackdata); }

    virtual int setcallbackalt(double (*ncallbackalt)(const double *, int, void *), void *ncallbackfndata) { return getBLK().setcallbackalt(ncallbackalt,ncallbackfndata); }

    virtual int setK0callbackalt(double (*nK0callbackalt)(int,                                                                                                         int, int, int, int, void *), void *nK0callbackdata) { return getBLK().setK0callbackalt(nK0callbackalt,nK0callbackdata); }
    virtual int setK1callbackalt(double (*nK1callbackalt)(int, const double *, int, int,                                                                               int, int, int, int, void *), void *nK1callbackdata) { return getBLK().setK1callbackalt(nK1callbackalt,nK1callbackdata); }
    virtual int setK2callbackalt(double (*nK2callbackalt)(int, const double *, int, const double *, int, int, int,                                                     int, int, int, int, void *), void *nK2callbackdata) { return getBLK().setK2callbackalt(nK2callbackalt,nK2callbackdata); }
    virtual int setK3callbackalt(double (*nK3callbackalt)(int, const double *, int, const double *, int, const double *, int, int, int, int,                           int, int, int, int, void *), void *nK3callbackdata) { return getBLK().setK3callbackalt(nK3callbackalt,nK3callbackdata); }
    virtual int setK4callbackalt(double (*nK4callbackalt)(int, const double *, int, const double *, int, const double *, int, const double *, int, int, int, int, int, int, int, int, int, void *), void *nK4callbackdata) { return getBLK().setK4callbackalt(nK4callbackalt,nK4callbackdata); }
    virtual int setKmcallbackalt(double (*nKmcallbackalt)(int, const double **, const int *, const int *, int,                                                         int, int, int, int, void *), void *nKmcallbackdata) { return getBLK().setKmcallbackalt(nKmcallbackalt,nKmcallbackdata); }

    virtual int setKcallbackalt(double (*nKcallbackalt)(int, double, double, int, int, int, int, void *), void *nKcallbackdata) { return getBLK().setKcallbackalt(nKcallbackalt,nKcallbackdata); }

#ifdef SWIG
#define KCALLALTS
#endif

#ifdef MAKENUMPYCOMP
#define KCALLALTS
#endif

#ifdef KCALLALTS
    virtual int setcallbackalt(PyObject *pycallback);

    virtual int setK0callbackalt(PyObject *pycallback);
    virtual int setK1callbackalt(PyObject *pycallback);
    virtual int setK2callbackalt(PyObject *pycallback);
    virtual int setK3callbackalt(PyObject *pycallback);
    virtual int setK4callbackalt(PyObject *pycallback);
    virtual int setKmcallbackalt(PyObject *pycallback);

    virtual int setKcallbackalt(PyObject *pycallback);
#endif

    virtual gcallback callback(void)   const { return getBLKconst().callback(); }

    virtual K0callbackfnalt K0callbackalt(void) const { return getBLKconst().K0callbackalt(); }
    virtual K1callbackfnalt K1callbackalt(void) const { return getBLKconst().K1callbackalt(); }
    virtual K2callbackfnalt K2callbackalt(void) const { return getBLKconst().K2callbackalt(); }
    virtual K3callbackfnalt K3callbackalt(void) const { return getBLKconst().K3callbackalt(); }
    virtual K4callbackfnalt K4callbackalt(void) const { return getBLKconst().K4callbackalt(); }
    virtual Kmcallbackfnalt Kmcallbackalt(void) const { return getBLKconst().Kmcallbackalt(); }

    virtual Kcallbackfnalt Kcallbackalt(void) const { return getBLKconst().Kcallbackalt(); }

    virtual void *callbackfndata(void) const { return getBLKconst().callbackfndata(); }

    virtual void *K0callbackdata(void) const { return getBLKconst().K0callbackdata(); }
    virtual void *K1callbackdata(void) const { return getBLKconst().K1callbackdata(); }
    virtual void *K2callbackdata(void) const { return getBLKconst().K2callbackdata(); }
    virtual void *K3callbackdata(void) const { return getBLKconst().K3callbackdata(); }
    virtual void *K4callbackdata(void) const { return getBLKconst().K4callbackdata(); }
    virtual void *Kmcallbackdata(void) const { return getBLKconst().Kmcallbackdata(); }

    // Callback string used by MEX interface

    virtual int setmexcall  (const std::string &xmexfn) { return getBLK().setmexcall(xmexfn);     }
    virtual int setmexcallid(int xmexfnid)              { return getBLK().setmexcallid(xmexfnid); }
    virtual const std::string &getmexcall  (void) const { return getBLKconst().getmexcall();      }
    virtual int                getmexcallid(void) const { return getBLKconst().getmexcallid();    }

    // System call stuff

    virtual int setsyscall(const std::string &xsysfn)   { return getBLK().setsyscall(xsysfn);   }
    virtual int setxfilename(const std::string &fname)  { return getBLK().setxfilename(fname);  }
    virtual int setyfilename(const std::string &fname)  { return getBLK().setyfilename(fname);  }
    virtual int setxyfilename(const std::string &fname) { return getBLK().setxyfilename(fname); }
    virtual int setyxfilename(const std::string &fname) { return getBLK().setyxfilename(fname); }
    virtual int setrfilename(const std::string &fname)  { return getBLK().setrfilename(fname);  }

    virtual const std::string &getsyscall(void)    const { return getBLKconst().getsyscall();    }
    virtual const std::string &getxfilename(void)  const { return getBLKconst().getxfilename();  }
    virtual const std::string &getyfilename(void)  const { return getBLKconst().getyfilename();  }
    virtual const std::string &getxyfilename(void) const { return getBLKconst().getxyfilename(); }
    virtual const std::string &getyxfilename(void) const { return getBLKconst().getyxfilename(); }
    virtual const std::string &getrfilename(void)  const { return getBLKconst().getrfilename();  }

    // BLK cache options

    virtual int mercachesize(void) const { return getBLKconst().mercachesize(); }
    virtual int setmercachesize(int nv) { return getBLK().setmercachesize(nv); }

    virtual int mercachenorm(void) const { return getBLKconst().mercachenorm(); }
    virtual int setmercachenorm(int nv) { return getBLK().setmercachenorm(nv); }

    // ML block averaging: set/remove element in list of ML blocks being averaged

    virtual int setmlqlist(int i, ML_Base &src)          { return getBLK().setmlqlist(i,src); }
    virtual int setmlqlist(const Vector<ML_Base *> &src) { return getBLK().setmlqlist(src);   }

    virtual int setmlqweight(int i, const gentype &w)  { return getBLK().setmlqweight(i,w); }
    virtual int setmlqweight(const Vector<gentype> &w) { return getBLK().setmlqweight(w);   }

    virtual int removemlqlist(int i) { return getBLK().removemlqlist(i); }

    const SparseVector<ML_Base *> mlqlist(void) const { return getBLKconst().mlqlist(); }
    const SparseVector<gentype>   mlqweight(void) const { return getBLKconst().mlqweight(); }

    // Kernel training:

//FIXME: ADDHERE
    virtual double minstepKB(void) const { return getBLKconst().minstepKB(); }
    virtual int    maxiterKB(void) const { return getBLKconst().maxiterKB(); }
    virtual double lrKB(void)      const { return getBLKconst().lrKB();      }

    virtual const Vector<int>    altMLidsKB(void) const { return getBLKconst().altMLidsKB(); }
    virtual const Vector<double> MLweightKB(void) const { return getBLKconst().MLweightKB(); }
    virtual const Matrix<double> &lambdaKB (void) const { return getBLKconst().lambdaKB();   }

    virtual int setminstepKB(double nv) { return getBLK().setminstepKB(nv); }
    virtual int setmaxiterKB(int    nv) { return getBLK().setmaxiterKB(nv); }
    virtual int setlrKB     (double nv) { return getBLK().setlrKB(nv);      }

    virtual int setaltMLidsKB(const Vector<int>    &nv) { return getBLK().setaltMLidsKB(nv); }
    virtual int setMLweightKB(const Vector<double> &nv) { return getBLK().setMLweightKB(nv); }
    virtual int setlambdaKB  (const Matrix<double> &nv) { return getBLK().setlambdaKB(nv);   }

    // Bernstein polynomials

//FIXME: ADDHERE
    virtual const gentype &bernDegree(void) const { return getBLKconst().bernDegree(); }
    virtual const gentype &bernIndex(void)  const { return getBLKconst().bernIndex();  }

    virtual int setBernDegree(const gentype &nv) { return getBLK().setBernDegree(nv); }
    virtual int setBernIndex(const gentype &nv)  { return getBLK().setBernIndex(nv);  }

//Variants
    virtual int setBernDegreeDbl(double nv) { gentype nnv(nv); return getBLK().setBernDegree(nnv); }
    virtual int setBernIndexDbl(double nv)  { gentype nnv(nv); return getBLK().setBernIndex(nnv);  }

//Variants
    virtual int setBernDegreeInt(int nv) { gentype nnv(nv); return getBLK().setBernDegree(nnv); }
    virtual int setBernIndexInt(int nv)  { gentype nnv(nv); return getBLK().setBernIndex(nnv);  }

    // Battery modelling parameters

    virtual const Vector<double> &battparam(void)      const { return getBLKconst().battparam();      }
    virtual const double         &batttmax(void)       const { return getBLKconst().batttmax();       }
    virtual const double         &battImax(void)       const { return getBLKconst().battImax();       }
    virtual const double         &batttdelta(void)     const { return getBLKconst().batttdelta();     }
    virtual const double         &battVstart(void)     const { return getBLKconst().battVstart();     }
    virtual const double         &battthetaStart(void) const { return getBLKconst().battthetaStart(); }

//Variants
    virtual void npbattparam(double **res, int *dim) const { *dim = battparam().size(); *res = const_cast<double *>(&(battparam()(zeroint()))); return; }

//FIXME: ADDHERE
    virtual int setbattparam(const Vector<gentype> &nv) { return getBLK().setbattparam(nv);      }
    virtual int setbatttmax(double nv)                  { return getBLK().setbatttmax(nv);       }
    virtual int setbattImax(double nv)                  { return getBLK().setbattImax(nv);       }
    virtual int setbatttdelta(double nv)                { return getBLK().setbatttdelta(nv);     }
    virtual int setbattVstart(double nv)                { return getBLK().setbattVstart(nv);     }
    virtual int setbattthetaStart(double nv)            { return getBLK().setbattthetaStart(nv); }







    // ================================================================
    //     Common functions for all KNNs
    // ================================================================

#ifndef SWIG
    virtual       KNN_Generic &getKNN     (void)       { NiceAssert( ( type() >= 300 ) && ( type() <= 399 ) ); return (dynamic_cast<      KNN_Generic &>(getML     ().getML     ())).getKNN();      }
    virtual const KNN_Generic &getKNNconst(void) const { NiceAssert( ( type() >= 300 ) && ( type() <= 399 ) ); return (dynamic_cast<const KNN_Generic &>(getMLconst().getMLconst())).getKNNconst(); }
#endif

    // Information functions (training data):

    virtual int k  (void) const { return getKNNconst().k  (); }
    virtual int ktp(void) const { return getKNNconst().ktp(); }

    // General modification and autoset functions

    virtual int setk  (int xk) { return getKNN().setk  (xk); }
    virtual int setktp(int xk) { return getKNN().setktp(xk); }









    // ================================================================
    //     Common functions for all GPs
    // ================================================================

#ifndef SWIG
    virtual       GPR_Generic &getGPR     (void)       { NiceAssert( ( type() >= 400 ) && ( type() <= 499 ) ); return (dynamic_cast<      GPR_Generic &>(getML     ().getML     ())).getGPR();      }
    virtual const GPR_Generic &getGPRconst(void) const { NiceAssert( ( type() >= 400 ) && ( type() <= 499 ) ); return (dynamic_cast<const GPR_Generic &>(getMLconst().getMLconst())).getGPRconst(); }
#endif

    // General modification and autoset functions

//FIXME: ADDHERE
    virtual int setmuWeight(const Vector<gentype> &nv) { return getGPR().setmuWeight(nv); }
    virtual int setmuBias  (const gentype         &nv) { return getGPR().setmuBias(nv);   }

//FIXME: ADDHERE
    virtual const Vector<gentype> &muWeight(void) const { return getGPRconst().muWeight(); }
    virtual const gentype         &muBias  (void) const { return getGPRconst().muBias();   }

    virtual int isZeromuBias(void) const { return getGPRconst().isZeromuBias(); }
    virtual int isVarmuBias (void) const { return getGPRconst().isVarmuBias();  }

    virtual int setZeromuBias(void) { return getGPR().setZeromuBias(); }
    virtual int setVarmuBias (void) { return getGPR().setVarmuBias();  }

    virtual const Matrix<double> &gprGp(void) { return getGPR().gprGp(); }

    // Likelihood

    virtual double loglikelihood(void) const { return getGPRconst().loglikelihood(); }













    // ================================================================
    //     Common functions for all LS-SVMs
    // ================================================================

#ifndef SWIG
    virtual       LSV_Generic &getLSV     (void)       { NiceAssert( ( type() >= 500 ) && ( type() <= 599 ) ); return (dynamic_cast<      LSV_Generic &>(getML     ().getML     ())).getLSV();      }
    virtual const LSV_Generic &getLSVconst(void) const { NiceAssert( ( type() >= 500 ) && ( type() <= 599 ) ); return (dynamic_cast<const LSV_Generic &>(getMLconst().getMLconst())).getLSVconst(); }
#endif

    // Constructors, destructors, assignment etc..

//FIXME: ADDHERE
    virtual int setgamma(const Vector<gentype> &newW) { return getLSV().setgamma(newW); }
    virtual int setdelta(const gentype         &newB) { return getLSV().setdelta(newB); }

    // Additional information

    virtual int isVardelta (void) const { return getLSVconst().isVardelta (); }
    virtual int isZerodelta(void) const { return getLSVconst().isZerodelta(); }

//FIXME: ADDHERE
    virtual const Vector<gentype> &gamma(void) const { return getLSVconst().gamma(); }
    virtual const gentype         &delta(void) const { return getLSVconst().delta(); }

    virtual const Matrix<double> &lsvGp(void) const { return getLSVconst().lsvGp(); }

    // General modification and autoset functions

    virtual int setVardelta (void) { return getLSV().setVardelta (); }
    virtual int setZerodelta(void) { return getLSV().setZerodelta(); }

    // Likelihood

    virtual double lsvloglikelihood(void) const { return getLSVconst().lsvloglikelihood(); }















    // ================================================================
    //     Common to all IMPs
    // ================================================================

#ifndef SWIG
    virtual       IMP_Generic &getIMP     (void)       { NiceAssert( ( type() >= 600 ) && ( type() <= 699 ) ); return (dynamic_cast<      IMP_Generic &>(getML     ().getML     ())).getIMP();      }
    virtual const IMP_Generic &getIMPconst(void) const { NiceAssert( ( type() >= 600 ) && ( type() <= 699 ) ); return (dynamic_cast<const IMP_Generic &>(getMLconst().getMLconst())).getIMPconst(); }
#endif

    // Improvement functions.

//FIXME: ADDHERE
    virtual int imp(gentype &resi, const SparseVector<gentype> &xxmean, const gentype &xxvar) const { return getIMPconst().imp(resi,xxmean,xxvar); }

    virtual double zref     (void) const { return getIMPconst().zref();      }
    virtual int    ehimethod(void) const { return getIMPconst().ehimethod(); }
    virtual int    needdg   (void) const { return getIMPconst().needdg();    }
    virtual double hypervol (void) const { return getIMPconst().hypervol();  }

    // Modification function

    virtual int setzref     (double nv) { return getIMP().setzref     (nv); }
    virtual int setehimethod(int    nv) { return getIMP().setehimethod(nv); }

















    // ================================================================
    //     Common to all SSVs
    // ================================================================

#ifndef SWIG
    virtual       SSV_Generic &getSSV     (void)       { NiceAssert( ( type() >= 700 ) && ( type() <= 799 ) ); return (dynamic_cast<      SSV_Generic &>(getML     ().getML     ())).getSSV();      }
    virtual const SSV_Generic &getSSVconst(void) const { NiceAssert( ( type() >= 700 ) && ( type() <= 799 ) ); return (dynamic_cast<const SSV_Generic &>(getMLconst().getMLconst())).getSSVconst(); }
#endif

    // General information and control

    virtual int Nzs(void) const { return getSSVconst().Nzs(); }

//FIXME: ADDHERE
    virtual const Vector<gentype>                &beta  (void) const { return getSSVconst().beta  (); }
    virtual const gentype                        &b     (void) const { return getSSVconst().b     (); }
    virtual const Vector<SparseVector<gentype> > &z     (void) const { return getSSVconst().z     (); }
    virtual const SparseVector<double>           &zmin  (void) const { return getSSVconst().zmin  (); }
    virtual const SparseVector<double>           &zmax  (void) const { return getSSVconst().zmax  (); }
    virtual const Vector<int>                    &xstate(void) const { return getSSVconst().xstate(); }
    virtual const Vector<int>                    &xact  (void) const { return getSSVconst().xact  (); }
    virtual const Matrix<double>                 &M     (void) const { return getSSVconst().M     (); }
    virtual const Vector<double>                 &n     (void) const { return getSSVconst().n     (); }

    virtual int isQuadRegul(void) const { return getSSVconst().isQuadRegul(); }
    virtual int isLinRegul (void) const { return getSSVconst().isLinRegul (); }

    virtual double biasForce(void) const { return getSSVconst().biasForce  (); }
    virtual int anomalclass(void)  const { return getSSVconst().anomalclass(); }

    // Control functions

    virtual int setbeta(const Vector<gentype> &newBeta) { return getSSV().setbeta(newBeta); }
    virtual int setb   (const gentype         &newb   ) { return getSSV().setb   (newb   ); }

//FIXME: ADDHERE
    virtual int setbeta(const Vector<double> &newBeta) { return getSSV().setbeta(newBeta); }
    virtual int setb   (const double         &newb   ) { return getSSV().setb   (newb   ); }

    virtual int setNzs(int nv) { return getSSV().setNzs(nv); }

    virtual int setzmin(const SparseVector<double> &nv) { return getSSV().setzmin(nv); }
    virtual int setzmax(const SparseVector<double> &nv) { return getSSV().setzmax(nv); }

    virtual int setQuadRegul(void) { return getSSV().setQuadRegul(); }
    virtual int setLinRegul (void) { return getSSV().setLinRegul (); }

    virtual int setBiasForce(double nv) { return getSSV().setBiasForce(nv);  }
    virtual int setanomalclass(int n)   { return getSSV().setanomalclass(n); }

    // Training control (for outer loop)

    virtual double ssvlr(void)       const { return getSSVconst().ssvlr      (); }
    virtual double ssvmom(void)      const { return getSSVconst().ssvmom     (); }
    virtual double ssvtol(void)      const { return getSSVconst().ssvtol     (); }
    virtual double ssvovsc(void)     const { return getSSVconst().ssvovsc    (); }
    virtual int    ssvmaxitcnt(void) const { return getSSVconst().ssvmaxitcnt(); }
    virtual double ssvmaxtime(void)  const { return getSSVconst().ssvmaxtime (); }

    virtual int setssvlr(double nv)      { return getSSV().setssvlr      (nv); }
    virtual int setssvmom(double nv)     { return getSSV().setssvmom     (nv); }
    virtual int setssvtol(double nv)     { return getSSV().setssvtol     (nv); }
    virtual int setssvovsc(double nv)    { return getSSV().setssvovsc    (nv); }
    virtual int setssvmaxitcnt(int nv)   { return getSSV().setssvmaxitcnt(nv); }
    virtual int setssvmaxtime(double nv) { return getSSV().setssvmaxtime (nv); }







    // ================================================================
    //     Common functions for all MLMs
    // ================================================================

#ifndef SWIG
    virtual       MLM_Generic &getMLM     (void)       { NiceAssert( ( type() >= 800 ) && ( type() <= 899 ) ); return (dynamic_cast<      MLM_Generic &>(getML     ().getML     ())).getMLM();      }
    virtual const MLM_Generic &getMLMconst(void) const { NiceAssert( ( type() >= 800 ) && ( type() <= 899 ) ); return (dynamic_cast<const MLM_Generic &>(getMLconst().getMLconst())).getMLMconst(); }
#endif

    // Back-propogation control
    //
    // "C" is set above
    // regtype is regularisation type.  1 for 1-norm, 2 for 2-norm
    // lr is learning rate

    virtual int tsize(void) const { return getMLMconst().tsize(); }
    virtual int knum(void)  const { return getMLMconst().knum();  }

    virtual int    regtype(int l) const { return getMLMconst().regtype(l); }
    virtual double regC(int l)    const { return getMLMconst().regC(l);    }
    virtual double mlmlr(void)    const { return getMLMconst().mlmlr();    }
    virtual double diffstop(void) const { return getMLMconst().diffstop(); }
    virtual double lsparse(void)  const { return getMLMconst().lsparse(); }

    virtual const Matrix<double> &GGp(int l) const { return getMLMconst().GGp(l); }

    virtual int settsize(int nv) { return getMLM().settsize(nv); }
    virtual int setknum(int nv)  { return getMLM().setknum(nv);  }

    virtual int setregtype(int l, int nv) { return getMLM().setregtype(l,nv); }
    virtual int setregC(int l, double nv) { return getMLM().setregC(l,nv);    }
    virtual int setmlmlr(double nv)       { return getMLM().setmlmlr(nv);     }
    virtual int setdiffstop(double nv)    { return getMLM().setdiffstop(nv);  }
    virtual int setlsparse(double nv)     { return getMLM().setlsparse(nv);   }






private:

    int mlType;
    Vector<ML_Base *> theML;
    int mlind;

    void resizetheML(int newsize);

    int isdelable; // set 1 if theML can be deleted, 0 otherwise
    SparseVector<std::string> locinfstore;

    ML_Mutable *thisthis;
    ML_Mutable **thisthisthis;
};

#ifndef SWIG
inline ML_Mutable *&setident (ML_Mutable *&a) { throw("What does that mean"); return a; }
inline ML_Mutable *&setposate(ML_Mutable *&a) { return a; }
inline ML_Mutable *&setnegate(ML_Mutable *&a) { throw("I reject your reality and substitute my own"); return a; }
inline ML_Mutable *&setconj  (ML_Mutable *&a) { throw("Have a kipper"); return a; }
inline ML_Mutable *&setrand  (ML_Mutable *&a) { throw("Cool...."); return a; }
inline ML_Mutable *&postProInnerProd(ML_Mutable *&a) { return a; }
#endif


#ifndef SWIG
inline void qswap(ML_Mutable &a, ML_Mutable &b)
{
    a.qswapinternal(b);

    return;
}
#endif

#ifndef SWIG
inline void qswap(ML_Mutable *&a, ML_Mutable *&b)
{
    ML_Mutable *temp;

    temp = a;
    a = b;
    b = temp;

    return;
}
#endif

#ifndef SWIG
inline void ML_Mutable::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    ML_Mutable &b = dynamic_cast<ML_Mutable &>(bb);

    ML_Base::qswapinternal(b);

    qswap(mlType,b.mlType);
    qswap(theML ,b.theML );
    qswap(mlind ,b.mlind );

    qswap(isdelable  ,b.isdelable  );
    qswap(locinfstore,b.locinfstore);

    return;
/*
Old version: don't need this AFAICT and it makes things much too complicated.
    if ( bb.isMutable() )
    {
        // qswap actually makes sense here even if types don't match

        ML_Mutable &b = dynamic_cast<ML_Mutable &>(bb);

        qswap(mlType,b.mlType);
        qswap(theML ,b.theML );
        qswap(mlind ,b.mlind );
    }

    else
    {
        getML().qswapinternal(bb.getML());
    }

    return;
*/
}
#endif

#ifndef SWIG
inline void ML_Mutable::semicopy(const ML_Base &bb)
{
    getML().semicopy(bb);

    return;
}
#endif

#ifndef SWIG
ML_Base &assign(ML_Base **dest, const ML_Base *src, int onlySemiCopy = 0);
#endif

#ifndef SWIG
inline void ML_Mutable::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    if ( bb.isMutable() )
    {
        const ML_Mutable &b = dynamic_cast<const ML_Mutable &>(bb);

        resizetheML((b.theML).size());

        if ( theML.size() )
        {
            int i;

            for ( i = 0 ; i < theML.size() ; i++ )
            {
                ::assign(&(theML("&",i)),&(b.getMLconst()),onlySemiCopy);
            }
        }

        mlind  = b.mlind;
        mlType = b.mlType;

        isdelable   = b.isdelable;
        locinfstore = b.locinfstore;
    }

    else
    {
        getML().assign(bb.getMLconst(),onlySemiCopy);
    }


    return;
}
#endif





// -----------------------------------------------------------------------
//
// Helper functions
//
// -----------------------------------------------------------------------

//
// Type list:
//
// Types: -2 = kernel precursor (not an ML)
//        -1 = none (base class constructed for some reason)
//         0 = Scalar SVM
//         1 = Binary SVM
//         2 = 1-class SVM
//         3 = Multiclass SVM
//         4 = Vector SVM
//         5 = Anionic SVM (real, complex, quaternion, octonion)
//         6 = auto-encoding SVM
//         7 = Density estimation SVM
//         8 = Pareto frontier SVM
//        12 = Binary score SVM
//        13 = Scalar Regression Score SVM
//        14 = Vector Regression Score SVM
//        15 = Generic target SVM
//        16 = planar SVM
//        17 = multi-expert rank SVM
//        18 = multi-user binary SVM
//        19 = similarity-learning SVM
//       100 = Scalar ONN
//       101 = Vector ONN
//       102 = Anion ONN
//       103 = Binary ONN
//       104 = Auto-encoding ONN
//       105 = Generic target ONN
//       200 = NOP machine (do nothing)
//       201 = Consensus machine (voting)
//       202 = Scalar average machine
//       203 = User-defined function (elementwise, g_i(x_i))
//       204 = User-I/O function
//       205 = Vector average machine
//       206 = Anion average machine
//       207 = User-defined function (vectorwise, g(x_0,x_1,...))
//       208 = Funcion callback machine
//       209 = Mex-defined function (elementwise, g_i(x_i))
//       210 = Mex-defined function (vectorwise, g(x_0,x_1,...))
//       211 = Mercer kernel inheritance block
//       212 = Multi ML averaging block
//       213 = system call block
//       214 = kernel block
//       215 = Bernstein kernel block
//       300 = KNN density estimator
//       301 = KNN binary classifier
//       302 = KNN generic regression
//       303 = KNN scalar regression
//       304 = KNN vector regression
//       305 = KNN anionic regression
//       306 = KNN autoencoder
//       307 = KNN multiclass classifier
//       400 = Scalar GP
//       401 = Vector GP
//       402 = Anion GP
//       408 = Generic target GP
//       409 = Binary classification GP
//       500 = Scalar LSV (LS-SVM)
//       501 = Vector LSV (LS-SVM)
//       502 = Anion LSV (LS-SVM)
//       505 = Scalar Regression Scoring LSV (LS-SVM)
//       506 = Vector Regression Scoring LSV (LS-SVM)
//       507 = auto-encoding LSV (LS-SVM)
//       508 = Generic target LSV (LS-SVM)
//       509 = Planar LSV (LS-SVM)
//       510 = Multi-expert rank LSV (LS-SVM)
//       600 = expected improvement (EI) IMP
//       601 = Pareto SVM 1-norm 1-class mono-surrogate
//       700 = SSV scalar regression
//       701 = SSV binary
//       701 = SSV 1-class
//       800 = SSV scalar regression
//
// Type ranges:      -2 kernel precursor (not an ML)
//                   -1 base type (not a functional ML)
//                0- 99 support vector machine (SVM)
//              100-199 one-layer layer neural network (ONN)
//              200-299 blocks (BLK)
//              300-399 k-nearest-neighbour machines (KNN)
//              400-499 Gaussian processes (GP)
//              500-599 Least-squares support vector machine (LSV)
//              600-699 Improvement measures (IMP)
//              700-799 Super-sparse support vector machine (SSV)
//              800-899 Type-II multi-layer kernel-machine (MLM)
//

//
// Generic machine learning block constructors.  Make ML of the given type
// (and subtype if specified).
//

ML_Base *makeNewML(int type, int subtype = -42);

//
// Identify type: takes first string read from file, converts to type
//

int convIDToType(const std::string &idstring);
int convTypeToID(std::string &idstringres, int id);

//
// Specific type identifiers
//

inline int isML(const ML_Base &src) { return ( src.type() >= 0 ); }

inline int isSVM(const ML_Base &src) { return ( src.type() >=   0 ) &&  ( src.type() <=  99 ); }
inline int isONN(const ML_Base &src) { return ( src.type() >= 100 ) &&  ( src.type() <= 199 ); }
inline int isBLK(const ML_Base &src) { return ( src.type() >= 200 ) &&  ( src.type() <= 299 ); }
inline int isKNN(const ML_Base &src) { return ( src.type() >= 300 ) &&  ( src.type() <= 399 ); }
inline int isGPR(const ML_Base &src) { return ( src.type() >= 400 ) &&  ( src.type() <= 499 ); }
inline int isLSV(const ML_Base &src) { return ( src.type() >= 500 ) &&  ( src.type() <= 599 ); }
inline int isIMP(const ML_Base &src) { return ( src.type() >= 600 ) &&  ( src.type() <= 699 ); }
inline int isSSV(const ML_Base &src) { return ( src.type() >= 700 ) &&  ( src.type() <= 799 ); }
inline int isMLM(const ML_Base &src) { return ( src.type() >= 800 ) &&  ( src.type() <= 899 ); }

inline int isSVMScalar(const ML_Base &src) { return ( src.type() ==   0 ); }
inline int isSVMBinary(const ML_Base &src) { return ( src.type() ==   1 ); }
inline int isSVMSingle(const ML_Base &src) { return ( src.type() ==   2 ); }
inline int isSVMMultiC(const ML_Base &src) { return ( src.type() ==   3 ); }
inline int isSVMVector(const ML_Base &src) { return ( src.type() ==   4 ); }
inline int isSVMAnions(const ML_Base &src) { return ( src.type() ==   5 ); }
inline int isSVMAutoEn(const ML_Base &src) { return ( src.type() ==   6 ); }
inline int isSVMDensit(const ML_Base &src) { return ( src.type() ==   7 ); }
inline int isSVMPFront(const ML_Base &src) { return ( src.type() ==   8 ); }
inline int isSVMBiScor(const ML_Base &src) { return ( src.type() ==  12 ); }
inline int isSVMScScor(const ML_Base &src) { return ( src.type() ==  13 ); }
inline int isSVMGentyp(const ML_Base &src) { return ( src.type() ==  15 ); }
inline int isSVMPlanar(const ML_Base &src) { return ( src.type() ==  16 ); }
inline int isSVMMvRank(const ML_Base &src) { return ( src.type() ==  17 ); }
inline int isSVMMulBin(const ML_Base &src) { return ( src.type() ==  18 ); }
inline int isSVMSimLrn(const ML_Base &src) { return ( src.type() ==  19 ); }
inline int isSVMCyclic(const ML_Base &src) { return ( src.type() ==  20 ); }

inline int isONNScalar(const ML_Base &src) { return ( src.type() == 100 ); }
inline int isONNVector(const ML_Base &src) { return ( src.type() == 101 ); }
inline int isONNAnions(const ML_Base &src) { return ( src.type() == 102 ); }
inline int isONNBinary(const ML_Base &src) { return ( src.type() == 103 ); }
inline int isONNAutoEn(const ML_Base &src) { return ( src.type() == 104 ); }
inline int isONNGentyp(const ML_Base &src) { return ( src.type() == 105 ); }

inline int isBLKNopnop(const ML_Base &src) { return ( src.type() == 200 ); }
inline int isBLKConsen(const ML_Base &src) { return ( src.type() == 201 ); }
inline int isBLKAveSca(const ML_Base &src) { return ( src.type() == 202 ); }
inline int isBLKUsrFnA(const ML_Base &src) { return ( src.type() == 203 ); }
inline int isBLKUserIO(const ML_Base &src) { return ( src.type() == 204 ); }
inline int isBLKAveVec(const ML_Base &src) { return ( src.type() == 205 ); }
inline int isBLKAveAni(const ML_Base &src) { return ( src.type() == 206 ); }
inline int isBLKUsrFnB(const ML_Base &src) { return ( src.type() == 207 ); }
inline int isBLKCalBak(const ML_Base &src) { return ( src.type() == 208 ); }
inline int isBLKMexFnA(const ML_Base &src) { return ( src.type() == 209 ); }
inline int isBLKMexFnB(const ML_Base &src) { return ( src.type() == 210 ); }
inline int isBLKMercer(const ML_Base &src) { return ( src.type() == 211 ); }
inline int isBLKConect(const ML_Base &src) { return ( src.type() == 212 ); }
inline int isBLKSystem(const ML_Base &src) { return ( src.type() == 213 ); }
inline int isBLKKernel(const ML_Base &src) { return ( src.type() == 214 ); }
inline int isBLKBernst(const ML_Base &src) { return ( src.type() == 215 ); }
inline int isBLKBatter(const ML_Base &src) { return ( src.type() == 216 ); }

inline int isKNNDensit(const ML_Base &src) { return ( src.type() == 300 ); }
inline int isKNNBinary(const ML_Base &src) { return ( src.type() == 301 ); }
inline int isKNNGentyp(const ML_Base &src) { return ( src.type() == 302 ); }
inline int isKNNScalar(const ML_Base &src) { return ( src.type() == 303 ); }
inline int isKNNVector(const ML_Base &src) { return ( src.type() == 304 ); }
inline int isKNNAnions(const ML_Base &src) { return ( src.type() == 305 ); }
inline int isKNNAutoEn(const ML_Base &src) { return ( src.type() == 306 ); }
inline int isKNNMultiC(const ML_Base &src) { return ( src.type() == 307 ); }

inline int isLSVScalar(const ML_Base &src) { return ( src.type() == 500 ); }
inline int isLSVVector(const ML_Base &src) { return ( src.type() == 501 ); }
inline int isLSVAnions(const ML_Base &src) { return ( src.type() == 502 ); }
inline int isLSVScScor(const ML_Base &src) { return ( src.type() == 505 ); }
inline int isLSVAutoEn(const ML_Base &src) { return ( src.type() == 507 ); }
inline int isLSVGentyp(const ML_Base &src) { return ( src.type() == 508 ); }
inline int isLSVPlanar(const ML_Base &src) { return ( src.type() == 509 ); }
inline int isLSVMvRank(const ML_Base &src) { return ( src.type() == 510 ); }

inline int isGPRScalar(const ML_Base &src) { return ( src.type() == 400 ); }
inline int isGPRVector(const ML_Base &src) { return ( src.type() == 401 ); }
inline int isGPRAnions(const ML_Base &src) { return ( src.type() == 402 ); }
inline int isGPRGentyp(const ML_Base &src) { return ( src.type() == 408 ); }
inline int isGPRBinary(const ML_Base &src) { return ( src.type() == 409 ); }

inline int isIMPExpect(const ML_Base &src) { return ( src.type() == 600 ); }
inline int isIMPParSVM(const ML_Base &src) { return ( src.type() == 601 ); }

inline int isSSVScalar(const ML_Base &src) { return ( src.type() == 700 ); }
inline int isSSVBinary(const ML_Base &src) { return ( src.type() == 701 ); }
inline int isSSVSingle(const ML_Base &src) { return ( src.type() == 702 ); }

inline int isMLMScalar(const ML_Base &src) { return ( src.type() == 800 ); }
inline int isMLMBinary(const ML_Base &src) { return ( src.type() == 801 ); }
inline int isMLMVector(const ML_Base &src) { return ( src.type() == 802 ); }

inline int isBinaryClassify(const ML_Base &src) { return isSVMBinary(src) || isSVMSingle(src) || isKNNBinary(src); }

#endif


