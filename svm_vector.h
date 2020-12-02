
//
// Vector regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _svm_vector_h
#define _svm_vector_h

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "svm_generic.h"
#include "svm_scalar.h"
#include "svm_vector_atonce.h"
#include "svm_vector_redbin.h"
#include "svm_vector_matonce.h"
#include "svm_vector_mredbin.h"




class SVM_Vector;
class LSV_Gentyp;


// Swap function

inline void qswap(SVM_Vector &a, SVM_Vector &b);


class SVM_Vector : public SVM_Generic
{
    friend class LSV_Gentyp;

public:

    // Constructors, destructors, assignment etc..

    SVM_Vector();
    SVM_Vector(const SVM_Vector &src);
    SVM_Vector(const SVM_Vector &src, const ML_Base *xsrc);
    SVM_Vector &operator=(const SVM_Vector  &src) { assign(src); return *this; }
    virtual ~SVM_Vector();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const;
    virtual void setmemsize(int memsize) { Qatonce.setmemsize(memsize); Qredbin.setmemsize(memsize); QMatonce.setmemsize(memsize); QMredbin.setmemsize(memsize); return; }

    virtual void assign(const ML_Base &src, int isOnlySemi = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);






    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    virtual       ML_Base &getML     (void)       { return static_cast<      ML_Base &>(getSVM());      }
    virtual const ML_Base &getMLconst(void) const { return static_cast<const ML_Base &>(getSVMconst()); }

    // Information functions (training data):

    virtual int N(void)       const { return getQconst().N  ();     }
    virtual int NNC(int d)    const { return getQconst().NNC(d);    }
    virtual int type(void)    const { return getQconst().type();    }
    virtual int subtype(void) const { return getQconst().subtype(); }

    virtual int tspaceDim(void)    const { return getQconst().tspaceDim();    }
    virtual int xspaceDim(void)    const { return getQconst().xspaceDim();    }
    virtual int fspaceDim(void)    const { return getQconst().fspaceDim();    }
    virtual int tspaceSparse(void) const { return getQconst().tspaceSparse(); }
    virtual int xspaceSparse(void) const { return getQconst().xspaceSparse(); }
    virtual int numClasses(void)   const { return getQconst().numClasses();   }
    virtual int order(void)        const { return getQconst().order();        }

    virtual int isTrained(void) const { return getQconst().isTrained(); }
    virtual int isMutable(void) const { return getQconst().isMutable(); }
    virtual int isPool   (void) const { return getQconst().isPool   (); }

    virtual char gOutType(void) const { return getQconst().gOutType(); }
    virtual char hOutType(void) const { return getQconst().hOutType(); }
    virtual char targType(void) const { return getQconst().targType(); }
    virtual double calcDist(const gentype &ha, const gentype &hb, int ia = -1, int db = 2) const { return getQconst().calcDist(ha,hb,ia,db); }

    virtual double calcDistInt(int    ha, int    hb, int ia = -1, int db = 2) const { return ML_Base::calcDistInt(ha,hb,ia,db); }
    virtual double calcDistDbl(double ha, double hb, int ia = -1, int db = 2) const { return ML_Base::calcDistDbl(ha,hb,ia,db); }

    virtual int isUnderlyingScalar(void) const { return getQconst().isUnderlyingScalar(); }
    virtual int isUnderlyingVector(void) const { return getQconst().isUnderlyingVector(); }
    virtual int isUnderlyingAnions(void) const { return getQconst().isUnderlyingAnions(); }

    virtual const Vector<int> &ClassLabels(void)   const { return getQconst().ClassLabels();        }
    virtual int getInternalClass(const gentype &y) const { return getQconst().getInternalClass(y);  }
    virtual int numInternalClasses(void)           const { return getQconst().numInternalClasses(); }
    virtual int isenabled(int i)                   const { return getQconst().isenabled(i);         }

    virtual const int *ClassLabelsInt(void) const { return ML_Base::ClassLabelsInt();       }
    virtual int  getInternalClassInt(int y) const { return ML_Base::getInternalClassInt(y); }

    virtual double C(void)         const { return getQconst().C();            }
    virtual double sigma(void)     const { return getQconst().sigma();        }
    virtual double eps(void)       const { return getQconst().eps();          }
    virtual double Cclass(int d)   const { return getQconst().Cclass(d);      }
    virtual double epsclass(int d) const { return getQconst().epsclass(d);    }

    virtual int    memsize(void)      const { return getQconst().memsize();      }
    virtual double zerotol(void)      const { return getQconst().zerotol();      }
    virtual double Opttol(void)       const { return getQconst().Opttol();       }
    virtual int    maxitcnt(void)     const { return getQconst().maxitcnt();     }
    virtual double maxtraintime(void) const { return getQconst().maxtraintime(); }

    virtual int    maxitermvrank(void) const { return getQconst().maxitermvrank(); }
    virtual double lrmvrank(void)      const { return getQconst().lrmvrank();      }
    virtual double ztmvrank(void)      const { return getQconst().ztmvrank();      }

    virtual double betarank(void) const { return getQconst().betarank(); }

    virtual double sparlvl(void) const { return getQconst().sparlvl(); }

    virtual const Vector<SparseVector<gentype> > &x          (void) const { return getQconst().x();           }
    virtual const Vector<gentype>                &y          (void) const { return getQconst().y();           }
    virtual const Vector<vecInfo>                &xinfo      (void) const { return getQconst().xinfo();       }
    virtual const Vector<int>                    &xtang      (void) const { return getQconst().xtang();       }
    virtual const Vector<int>                    &d          (void) const { return getQconst().d();           }
    virtual const Vector<double>                 &Cweight    (void) const { return getQconst().Cweight();     }
    virtual const Vector<double>                 &Cweightfuzz(void) const { return getQconst().Cweightfuzz(); }
    virtual const Vector<double>                 &sigmaweight(void) const { return getQconst().sigmaweight(); }
    virtual const Vector<double>                 &epsweight  (void) const { return getQconst().epsweight();   }
    virtual const Vector<int>                    &alphaState (void) const { return getQconst().alphaState();  }

    virtual void npCweight    (double **res, int *dim) const { ML_Base::npCweight    (res,dim); return; }
    virtual void npCweightfuzz(double **res, int *dim) const { ML_Base::npCweightfuzz(res,dim); return; }
    virtual void npsigmaweight(double **res, int *dim) const { ML_Base::npsigmaweight(res,dim); return; }
    virtual void npepsweight  (double **res, int *dim) const { ML_Base::npepsweight  (res,dim); return; }

    virtual int isClassifier(void) const { return getQconst().isClassifier(); }
    virtual int isRegression(void) const { return getQconst().isRegression(); }

    // Version numbers

    virtual int xvernum(void)        const { return getQconst().xvernum();        }
    virtual int xvernum(int altMLid) const { return getQconst().xvernum(altMLid); }
    virtual int incxvernum(void)           { return getQ().incxvernum();          }
    virtual int gvernum(void)        const { return getQconst().gvernum();        }
    virtual int gvernum(int altMLid) const { return getQconst().gvernum(altMLid); }
    virtual int incgvernum(void)           { return getQ().incgvernum();          }

    virtual int MLid(void) const { return getQconst().MLid(); }
    virtual int setMLid(int nv) { return getQ().setMLid(nv); }
    virtual int getaltML(kernPrecursor *&res, int altMLid) const { return getQconst().getaltML(res,altMLid); }

    // RKHS inner-product support

    virtual void mProdPt(double &res, int m, int *x) { getQ().mProdPt(res,m,x); return; }

    // Kernel Modification

    virtual const MercerKernel &getKernel(void) const                                           { return getQconst().getKernel();        }
    virtual MercerKernel &getKernel_unsafe(void)                                                { return getQ().getKernel_unsafe();      }
    virtual void prepareKernel(void)                                                            {        getQ().prepareKernel(); return; }
    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1)        { int res = Qatonce.resetKernel(modind,onlyChangeRowI,updateInfo); res |= Qredbin.resetKernel(modind,onlyChangeRowI,updateInfo); return res; }
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1) { int res = Qatonce.setKernel(xkernel,modind,onlyChangeRowI);      res |= Qredbin.setKernel(xkernel,modind,onlyChangeRowI);      return res; }

    virtual void fillCache(void) { getQ().fillCache(); return; }

    virtual void K2bypass(const Matrix<gentype> &nv) { getQ().K2bypass(nv); return; }

    virtual gentype &Keqn(gentype &res,                           int resmode = 1) const { return getQconst().Keqn(res,     resmode); }
    virtual gentype &Keqn(gentype &res, const MercerKernel &altK, int resmode = 1) const { return getQconst().Keqn(res,altK,resmode); }

    virtual gentype &K1(gentype &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().K1(res,xa,xainf); }
    virtual gentype &K2(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().K2(res,xa,xb,xainf,xbinf); }
    virtual gentype &K3(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL) const { return getQconst().K3(res,xa,xb,xc,xainf,xbinf,xcinf); }
    virtual gentype &K4(gentype &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, const vecInfo *xcinf = NULL, const vecInfo *xdinf = NULL) const { return getQconst().K4(res,xa,xb,xc,xd,xainf,xbinf,xcinf,xdinf); }
    virtual gentype &Km(gentype &res, const Vector<SparseVector<gentype> > &xx) const { return getQconst().Km(res,xx); }

    virtual double &K2ip(double &res, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().K2ip(res,xa,xb,xainf,xbinf); }
    virtual double distK(const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL) const { return getQconst().distK(xa,xb,xainf,xbinf); }

    virtual Vector<gentype> &phi2(Vector<gentype> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<gentype> &phi2(Vector<gentype> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,ia,xa,xainf); }

    virtual Vector<double> &phi2(Vector<double> &res, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,xa,xainf); }
    virtual Vector<double> &phi2(Vector<double> &res, int ia, const SparseVector<gentype> *xa = NULL, const vecInfo *xainf = NULL) const { return getQconst().phi2(res,ia,xa,xainf); }

    virtual double &K0ip(       double &res, const gentype **pxyprod = NULL) const { return getQconst().K0ip(res,pxyprod); }
    virtual double &K1ip(       double &res, int i, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const vecInfo *xxinfo = NULL) const { return  getQconst().K1ip(res,i,pxyprod,xx,xxinfo); }
    virtual double &K2ip(       double &res, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { return getQconst().K2ip(res,i,j,pxyprod,xx,yy,xxinfo,yyinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return getQconst().K3ip(res,ia,ib,ic,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return getQconst().K4ip(res,ia,ib,ic,id,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return getQconst().Kmip(m,res,i,pxyprod,xx,xxinfo); }

    virtual double &K0ip(       double &res, const double &bias, const gentype **pxyprod = NULL) const { return getQconst().K0ip(res,bias,pxyprod); }
    virtual double &K1ip(       double &res, int ia, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL) const { return getQconst().K1ip(res,ia,bias,pxyprod,xa,xainfo); }
    virtual double &K2ip(       double &res, int ia, int ib, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL) const { return getQconst().K2ip(res,ia,ib,bias,pxyprod,xa,xb,xainfo,xbinfo); }
    virtual double &K3ip(       double &res, int ia, int ib, int ic, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL) const { return getQconst().K3ip(res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo); }
    virtual double &K4ip(       double &res, int ia, int ib, int ic, int id, const double &bias, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL) const { return getQconst().K4ip(res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo); }
    virtual double &Kmip(int m, double &res, Vector<int> &i, const double &bias, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL) const { return getQconst().Kmip(m,res,i,bias,pxyprod,xx,xxinfo); }

    virtual gentype        &K0(              gentype        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const gentype &bias     , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res,bias,pxyprod,resmode); }
    virtual gentype        &K0(              gentype        &res, const MercerKernel &altK, const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res,altK,pxyprod,resmode); }
    virtual double         &K0(              double         &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(         res     ,pxyprod,resmode); }
    virtual Matrix<double> &K0(int spaceDim, Matrix<double> &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(spaceDim,res     ,pxyprod,resmode); }
    virtual d_anion        &K0(int order,    d_anion        &res                          , const gentype **pxyprod = NULL, int resmode = 0) const { return getQconst().K0(order   ,res     ,pxyprod,resmode); }

    virtual gentype        &K1(              gentype        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia,bias,pxyprod,xa,xainfo,resmode); }
    virtual gentype        &K1(              gentype        &res, int ia, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia,altK,pxyprod,xa,xainfo,resmode); }
    virtual double         &K1(              double         &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(         res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual Matrix<double> &K1(int spaceDim, Matrix<double> &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(spaceDim,res,ia     ,pxyprod,xa,xainfo,resmode); }
    virtual d_anion        &K1(int order,    d_anion        &res, int ia                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const vecInfo *xainfo = NULL, int resmode = 0) const { return getQconst().K1(order   ,res,ia     ,pxyprod,xa,xainfo,resmode); }

    virtual gentype        &K2(              gentype        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int i, int j, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,i,j,bias,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual gentype        &K2(              gentype        &res, int i, int j, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,i,j,altK,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual double         &K2(              double         &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(         res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual Matrix<double> &K2(int spaceDim, Matrix<double> &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(spaceDim,res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }
    virtual d_anion        &K2(int order,    d_anion        &res, int i, int j                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int resmode = 0) const { return getQconst().K2(order,   res,i,j     ,pxyprod,xx,yy,xxinfo,yyinfo,resmode); }

    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic,bias,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual gentype        &K3(              gentype        &res, int ia, int ib, int ic, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic,altK,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual double         &K3(              double         &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(         res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual Matrix<double> &K3(int spaceDim, Matrix<double> &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(spaceDim,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }
    virtual d_anion        &K3(int order,    d_anion        &res, int ia, int ib, int ic                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, int resmode = 0) const { return getQconst().K3(order   ,res,ia,ib,ic     ,pxyprod,xa,xb,xc,xainfo,xbinfo,xcinfo,resmode); }

    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const gentype &bias     , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id,bias,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual gentype        &K4(              gentype        &res, int ia, int ib, int ic, int id, const MercerKernel &altK, const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id,altK,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual double         &K4(              double         &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(         res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual Matrix<double> &K4(int spaceDim, Matrix<double> &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(spaceDim,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }
    virtual d_anion        &K4(int order,    d_anion        &res, int ia, int ib, int ic, int id                          , const gentype **pxyprod = NULL, const SparseVector<gentype> *xa = NULL, const SparseVector<gentype> *xb = NULL, const SparseVector<gentype> *xc = NULL, const SparseVector<gentype> *xd = NULL, const vecInfo *xainfo = NULL, const vecInfo *xbinfo = NULL, const vecInfo *xcinfo = NULL, const vecInfo *xdinfo = NULL, int resmode = 0) const { return getQconst().K4(order   ,res,ia,ib,ic,id     ,pxyprod,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,resmode); }

    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const gentype &bias     , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,bias,pxyprod,xx,xxinfo,resmode); }
    virtual gentype        &Km(int m              , gentype        &res, Vector<int> &i, const MercerKernel &altK, const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i,altK,pxyprod,xx,xxinfo,resmode); }
    virtual double         &Km(int m              , double         &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m         ,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual Matrix<double> &Km(int m, int spaceDim, Matrix<double> &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m,spaceDim,res,i     ,pxyprod,xx,xxinfo,resmode); }
    virtual d_anion        &Km(int m, int order   , d_anion        &res, Vector<int> &i                          , const gentype **pxyprod = NULL, Vector<const SparseVector<gentype> *> *xx = NULL, Vector<const vecInfo *> *xxinfo = NULL, int resmode = 0) const { return getQconst().Km(m,order   ,res,i     ,pxyprod,xx,xxinfo,resmode); }

    virtual void dK(gentype &xygrad, gentype &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }
    virtual void dK(double  &xygrad, double  &xnormgrad, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL, int deepDeriv = 0) const { getQconst().dK(xygrad,xnormgrad,i,j,     pxyprod,xx,yy,xxinfo,yyinfo,deepDeriv); return; }

    virtual void d2K(gentype &xygrad, gentype &xnormgrad, gentype &xyxygrad, gentype &xyxnormgrad, gentype &xyynormgrad, gentype &xnormxnormgrad, gentype &xnormynormgrad, gentype &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K(double  &xygrad, double  &xnormgrad, double  &xyxygrad, double  &xyxnormgrad, double  &xyynormgrad, double  &xnormxnormgrad, double  &xnormynormgrad, double  &ynormynormgrad, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K(xygrad,xnormgrad,xyxygrad,xyxnormgrad,xyynormgrad,xnormxnormgrad,xnormynormgrad,ynormynormgrad,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dK2delx(gentype &xscaleres, gentype &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dK2delx(double  &xscaleres, double  &yscaleres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dK2delx(xscaleres,yscaleres,minmaxind,i,j,     pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(gentype &xxscaleres, gentype &yyscaleres, gentype &xyscaleres, gentype &yxscaleres, gentype &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void d2K2delxdelx(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdelx(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void d2K2delxdely(double  &xxscaleres, double  &yyscaleres, double  &xyscaleres, double  &yxscaleres, double  &constres, int &minmaxind, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().d2K2delxdely(xxscaleres,yyscaleres,xyscaleres,yxscaleres,constres,minmaxind,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual void dnK2del(Vector<gentype> &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }
    virtual void dnK2del(Vector<double>  &sc, Vector<Vector<int> > &n, int &minmaxind, const Vector<int> &q, int i, int j, const gentype **pxyprod = NULL, const SparseVector<gentype> *xx = NULL, const SparseVector<gentype> *yy = NULL, const vecInfo *xxinfo = NULL, const vecInfo *yyinfo = NULL) const { getQconst().dnK2del(sc,n,minmaxind,q,i,j,pxyprod,xx,yy,xxinfo,yyinfo); return; }

    virtual double distK(int i, int j) const { return getQconst().distK(i,j); }

    virtual void densedKdx(double &res, int i, int j) const { return getQconst().densedKdx(res,i,j); }
    virtual void denseintK(double &res, int i, int j) const { return getQconst().denseintK(res,i,j); }

    virtual void densedKdx(double &res, int i, int j, const double &bias) const { return getQconst().densedKdx(res,i,j,bias); }
    virtual void denseintK(double &res, int i, int j, const double &bias) const { return getQconst().denseintK(res,i,j,bias); }

    virtual void ddistKdx(double &xscaleres, double &yscaleres, int &minmaxind, int i, int j) const { getQconst().ddistKdx(xscaleres,yscaleres,minmaxind,i,j); return; }

    virtual int isKVarianceNZ(void) const { return getQconst().isKVarianceNZ(); }

    virtual void K0xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, int xdim, int densetype, int resmode, int mlid) const { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(gentype &dxyprod, gentype &ddiffis, gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb,const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const{ getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                    gentype &res, int &minmaxind, int typeis, const gentype &xyprod, const gentype &yxprod, const gentype &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual void K0xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, int xdim, int densetype, int resmode, int mlid) const { getQconst().K0xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xdim,densetype,resmode,mlid); return; }
    virtual void K1xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const vecInfo &xainfo, int ia, int xdim, int densetype, int resmode, int mlid) const { getQconst().K1xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xainfo,ia,xdim,densetype,resmode,mlid); return; }
    virtual void K2xfer(double &dxyprod, double &ddiffis, double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo &xainfo, const vecInfo &xbinfo, int ia, int ib, int xdim, int densetype, int resmode, int mlid) const { getQconst().K2xfer(dxyprod,ddiffis,res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xainfo,xbinfo,ia,ib,xdim,densetype,resmode,mlid); return; }
    virtual void K3xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, int ia, int ib, int ic, int xdim, int densetype, int resmode, int mlid) const { getQconst().K3xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xainfo,xbinfo,xcinfo,ia,ib,ic,xdim,densetype,resmode,mlid); return; }
    virtual void K4xfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const SparseVector<gentype> &xc, const SparseVector<gentype> &xd, const vecInfo &xainfo, const vecInfo &xbinfo, const vecInfo &xcinfo, const vecInfo &xdinfo, int ia, int ib, int ic, int id, int xdim, int densetype, int resmode, int mlid) const { getQconst().K4xfer(res,minmaxind,typeis,xyprod,yxprod,diffis,xa,xb,xc,xd,xainfo,xbinfo,xcinfo,xdinfo,ia,ib,ic,id,xdim,densetype,resmode,mlid); return; }
    virtual void Kmxfer(                                  double &res, int &minmaxind, int typeis, const double &xyprod, const double &yxprod, const double &diffis, Vector<const SparseVector<gentype> *> &x, Vector<const vecInfo *> &xinfo, Vector<int> &i, int xdim, int m, int densetype, int resmode, int mlid) const { getQconst().Kmxfer(res,minmaxind,typeis,xyprod,yxprod,diffis,x,xinfo,i,xdim,m,densetype,resmode,mlid); return; }

    virtual const gentype &xelm(gentype &res, int i, int j) const { return getQconst().xelm(res,i,j); }
    virtual int xindsize(int i) const { return getQconst().xindsize(i); }

    // Training set modification:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector(int i,            double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) { return ML_Base::addTrainingVector(i,   xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, int zz,    double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }
    virtual int addTrainingVector(int i, double zz, double *xxa, int dima, double Cweigh = 1, double epsweigh = 1) { return ML_Base::addTrainingVector(i,zz,xxa,dima,Cweigh,epsweigh); }

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i)                                              { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x) { return getQ().removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, int num)                                     { return getQ().removeTrainingVector(i,num); }

    virtual int setx(int i, const SparseVector<gentype> &x)                         { return getQ().setx(i,x); }
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x) { return getQ().setx(i,x); }
    virtual int setx(const Vector<SparseVector<gentype> > &x)                       { return getQ().setx(x); }

    virtual int qswapx(int                i, SparseVector<gentype>          &x, int dontupdate = 0) { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(const Vector<int> &i, Vector<SparseVector<gentype> > &x, int dontupdate = 0) { return getQ().qswapx(i,x,dontupdate); }
    virtual int qswapx(                      Vector<SparseVector<gentype> > &x, int dontupdate = 0) { return getQ().qswapx(  x,dontupdate); }

    virtual int sety(int i, const gentype &z)                        { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<gentype> &z)                       { return getQ().sety(z); }

    virtual int sety(int                i, double                z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<double> &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<double> &z) { return getQ().sety(z); }

    virtual int sety(int                i, const Vector<double>          &z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<Vector<double> > &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<Vector<double> > &z) { return getQ().sety(z); }

    virtual int sety(int                i, const d_anion         &z) { return getQ().sety(i,z); }
    virtual int sety(const Vector<int> &i, const Vector<d_anion> &z) { return getQ().sety(i,z); }
    virtual int sety(                      const Vector<d_anion> &z) { return getQ().sety(z); }

    virtual int setd(int i, int d)                               { return getQ().setd(i,d); }
    virtual int setd(const Vector<int> &i, const Vector<int> &d) { return getQ().setd(i,d); }
    virtual int setd(const Vector<int> &d)                       { return getQ().setd(d); }

    virtual int setCweight(int i, double xCweight)                               { return getQ().setCweight(i,xCweight); }
    virtual int setCweight(const Vector<int> &i, const Vector<double> &xCweight) { return getQ().setCweight(i,xCweight); }
    virtual int setCweight(const Vector<double> &xCweight)                       { return getQ().setCweight(xCweight); }

    virtual int setCweightfuzz(int i, double xCweight)                               { return getQ().setCweightfuzz(i,xCweight); }
    virtual int setCweightfuzz(const Vector<int> &i, const Vector<double> &xCweight) { return getQ().setCweightfuzz(i,xCweight); }
    virtual int setCweightfuzz(const Vector<double> &xCweight)                       { return getQ().setCweightfuzz(xCweight); }

    virtual int setsigmaweight(int i, double xCweight)                               { return getQ().setsigmaweight(i,xCweight); }
    virtual int setsigmaweight(const Vector<int> &i, const Vector<double> &xCweight) { return getQ().setsigmaweight(i,xCweight); }
    virtual int setsigmaweight(const Vector<double> &xCweight)                       { return getQ().setsigmaweight(xCweight); }

    virtual int setepsweight(int i, double xepsweight)                               { return getQ().setepsweight(i,xepsweight); }
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight) { return getQ().setepsweight(i,xepsweight); }
    virtual int setepsweight(const Vector<double> &xepsweight)                       { return getQ().setepsweight(xepsweight); }

    virtual int scaleCweight(double scalefactor)     { return getQ().scaleCweight(scalefactor);     }
    virtual int scaleCweightfuzz(double scalefactor) { return getQ().scaleCweightfuzz(scalefactor); }
    virtual int scalesigmaweight(double scalefactor) { return getQ().scalesigmaweight(scalefactor); }
    virtual int scaleepsweight(double scalefactor)   { return getQ().scaleepsweight(scalefactor);   }

    virtual void assumeConsistentX  (void) { getQ().assumeConsistentX  (); return; }
    virtual void assumeInconsistentX(void) { getQ().assumeInconsistentX(); return; }

    virtual int isXConsistent(void)        const { return getQconst().isXConsistent();        }
    virtual int isXAssumedConsistent(void) const { return getQconst().isXAssumedConsistent(); }

    virtual void xferx(const ML_Base &xsrc) { getQ().xferx(xsrc); return; }

    virtual const vecInfo &xinfo(int i)            const { return getQconst().xinfo(i); }
    virtual int xtang(int i)                       const { return getQconst().xtang(i); }
    virtual const SparseVector<gentype> &x(int i)  const { return getQconst().x(i); }
    virtual int xisrank(int i)                     const { return getQconst().xisrank(i);  }
    virtual int xisgrad(int i)                     const { return getQconst().xisgrad(i);  }
    virtual int xisrankorgrad(int i)               const { return getQconst().xisrankorgrad(i);  }
    virtual int xisclass(int i, int d, int q = -1) const { return getQconst().xisclass(i,d,q); }
    virtual const gentype &y(int i)                const { return getQconst().y(i); }

    // Basis stuff

    virtual int NbasisUU(void)    const { return getQconst().NbasisUU();    }
    virtual int basisTypeUU(void) const { return getQconst().basisTypeUU(); }
    virtual int defProjUU(void)   const { return getQconst().defProjUU();   }

    virtual const Vector<gentype> &VbasisUU(void) const { return getQconst().VbasisUU(); }

    virtual int setBasisYUU(void)                     { return getQ().setBasisYUU();             }
    virtual int setBasisUUU(void)                     { return getQ().setBasisUUU();             }
    virtual int addToBasisUU(int i, const gentype &o) { return getQ().addToBasisUU(i,o);         }
    virtual int removeFromBasisUU(int i)              { return getQ().removeFromBasisUU(i);      }
    virtual int setBasisUU(int i, const gentype &o)   { return getQ().setBasisUU(i,o);           }
    virtual int setBasisUU(const Vector<gentype> &o)  { return getQ().setBasisUU(o);             }
    virtual int setDefaultProjectionUU(int d)         { return getQ().setDefaultProjectionUU(d); }
    virtual int setBasisUU(int n, int d)              { return getQ().setBasisUU(n,d);           }

    virtual int NbasisVV(void)    const { return getQconst().NbasisVV();    }
    virtual int basisTypeVV(void) const { return getQconst().basisTypeVV(); }
    virtual int defProjVV(void)   const { return getQconst().defProjVV();   }

    virtual const Vector<gentype> &VbasisVV(void) const { return getQconst().VbasisVV(); }

    virtual int setBasisYVV(void)                     { return getQ().setBasisYVV();             }
    virtual int setBasisUVV(void)                     { return getQ().setBasisUVV();             }
    virtual int addToBasisVV(int i, const gentype &o) { return getQ().addToBasisVV(i,o);         }
    virtual int removeFromBasisVV(int i)              { return getQ().removeFromBasisVV(i);      }
    virtual int setBasisVV(int i, const gentype &o)   { return getQ().setBasisVV(i,o);           }
    virtual int setBasisVV(const Vector<gentype> &o)  { return getQ().setBasisVV(o);             }
    virtual int setDefaultProjectionVV(int d)         { return getQ().setDefaultProjectionVV(d); }
    virtual int setBasisVV(int n, int d)              { return getQ().setBasisVV(n,d);           }

    virtual const MercerKernel &getUUOutputKernel(void) const                  { return getQconst().getUUOutputKernel();          }
    virtual MercerKernel &getUUOutputKernel_unsafe(void)                       { return getQ().getUUOutputKernel_unsafe();        }
    virtual int resetUUOutputKernel(int modind = 1)                            { return getQ().resetUUOutputKernel(modind);       }
    virtual int setUUOutputKernel(const MercerKernel &xkernel, int modind = 1) { return getQ().setUUOutputKernel(xkernel,modind); }

    // General modification and autoset functions

    virtual int randomise(double sparsity) { return getQ().randomise(sparsity); }
    virtual int autoen(void)               { return getQ().autoen();            }
    virtual int renormalise(void)          { return getQ().renormalise();       }
    virtual int realign(void)              { return getQ().realign();           }

    virtual int setzerotol(double zt)                 { int res = Qatonce.setzerotol(zt);                 res |= Qredbin.setzerotol(zt);                 res |= QMatonce.setzerotol(zt);                 res |= QMredbin.setzerotol(zt);                 return res; }
    virtual int setOpttol(double xopttol)             { int res = Qatonce.setOpttol(xopttol);             res |= Qredbin.setOpttol(xopttol);             res |= QMatonce.setOpttol(xopttol);             res |= QMredbin.setOpttol(xopttol);             return res; }
    virtual int setmaxitcnt(int xmaxitcnt)            { int res = Qatonce.setmaxitcnt(xmaxitcnt);         res |= Qredbin.setmaxitcnt(xmaxitcnt);         res |= QMatonce.setmaxitcnt(xmaxitcnt);         res |= QMredbin.setmaxitcnt(xmaxitcnt);         return res; }
    virtual int setmaxtraintime(double xmaxtraintime) { int res = Qatonce.setmaxtraintime(xmaxtraintime); res |= Qredbin.setmaxtraintime(xmaxtraintime); res |= QMatonce.setmaxtraintime(xmaxtraintime); res |= QMredbin.setmaxtraintime(xmaxtraintime); return res; }

    virtual int setmaxitermvrank(int nv) { return getQ().setmaxitermvrank(nv); }
    virtual int setlrmvrank(double nv)   { return getQ().setlrmvrank(nv);      }
    virtual int setztmvrank(double nv)   { return getQ().setztmvrank(nv);      }

    virtual int setbetarank(double nv) { return getQ().setbetarank(nv); }

    virtual int setC(double xC)                 { int res = Qatonce.setC(xC);            res |= Qredbin.setC(xC);            res |= QMatonce.setC(xC);            res |= QMredbin.setC(xC);            return res; }
    virtual int setsigma(double xC)             { int res = Qatonce.setsigma(xC);        res |= Qredbin.setsigma(xC);        res |= QMatonce.setsigma(xC);        res |= QMredbin.setsigma(xC);        return res; }
    virtual int seteps(double xeps)             { int res = Qatonce.seteps(xeps);        res |= Qredbin.seteps(xeps);        res |= QMatonce.seteps(xeps);        res |= QMredbin.seteps(xeps);        return res; }
    virtual int setCclass(int d, double xC)     { int res = Qatonce.setCclass(d,xC);     res |= Qredbin.setCclass(d,xC);     res |= QMatonce.setCclass(d,xC);     res |= QMredbin.setCclass(d,xC);     return res; }
    virtual int setepsclass(int d, double xeps) { int res = Qatonce.setepsclass(d,xeps); res |= Qredbin.setepsclass(d,xeps); res |= QMatonce.setepsclass(d,xeps); res |= QMredbin.setepsclass(d,xeps); return res; }

    virtual int scale(double a);
    virtual int reset(void);
    virtual int restart(void) { SVM_Vector temp; *this = temp; return 1; }
    virtual int home(void)    { return getQ().home(); }

    virtual ML_Base &operator*=(double sf) { scale(sf); return *this; }

    virtual int scaleby(double sf) { *this *= sf; return 1; }

    virtual int settspaceDim(int newdim) { int res = Qatonce.settspaceDim(newdim); res |= Qredbin.settspaceDim(newdim); res |= QMatonce.settspaceDim(newdim); res |= QMredbin.settspaceDim(newdim); return res; }
    virtual int addtspaceFeat(int i)     { int res = Qatonce.addtspaceFeat(i);     res |= Qredbin.addtspaceFeat(i);     res |= QMatonce.addtspaceFeat(i);     res |= QMredbin.addtspaceFeat(i);     return res; }
    virtual int removetspaceFeat(int i)  { int res = Qatonce.removetspaceFeat(i);  res |= Qredbin.removetspaceFeat(i);  res |= QMatonce.removetspaceFeat(i);  res |= QMredbin.removetspaceFeat(i);  return res; }
    virtual int addxspaceFeat(int i)     { int res = Qatonce.addxspaceFeat(i);     res |= Qredbin.addxspaceFeat(i);     res |= QMatonce.addxspaceFeat(i);     res |= QMredbin.addxspaceFeat(i);     return res; }
    virtual int removexspaceFeat(int i)  { int res = Qatonce.removexspaceFeat(i);  res |= Qredbin.removexspaceFeat(i);  res |= QMatonce.removexspaceFeat(i);  res |= QMredbin.removexspaceFeat(i);  return res; }

    virtual int setsubtype(int i);

    virtual int setorder(int neword)                 { int res = Qatonce.setorder(neword);         res |= Qredbin.setorder(neword);         res |= QMatonce.setorder(neword);         res |= QMredbin.setorder(neword);        return res; }
    virtual int addclass(int label, int epszero = 0) { int res = Qatonce.addclass(label,epszero);  res |= Qredbin.addclass(label,epszero);  res |= QMatonce.addclass(label,epszero);  res |= QMredbin.addclass(label,epszero); return res; }

    // Sampling mode

    virtual int isSampleMode(void) const { return getQconst().isSampleMode(); }
    virtual int setSampleMode(int nv, const Vector<gentype> &xmin, const Vector<gentype> &xmax, int Nsamp = DEFAULT_SAMPLES_SAMPLE, int sampSplit = 1, int sampType = 0) { return getQ().setSampleMode(nv,xmin,xmax,Nsamp,sampSplit,sampType); }

    // Training functions:

    virtual void fudgeOn(void)  { Qatonce.fudgeOn();  Qredbin.fudgeOn();  return; }
    virtual void fudgeOff(void) { Qatonce.fudgeOff(); Qredbin.fudgeOff(); return; }

    virtual int train(int &res) { return getQ().train(res); }
    virtual int train(int &res, svmvolatile int &killSwitch) { return getQ().train(res,killSwitch); }

    // Evaluation Functions:

    virtual int ggTrainingVector(               gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(     resg,i,retaltg,pxyprodi); }
    virtual int hhTrainingVector(gentype &resh,                int i,                  gentype ***pxyprodi = NULL) const { return getQconst().hhTrainingVector(resh,     i,        pxyprodi); }
    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ghTrainingVector(resh,resg,i,retaltg,pxyprodi); }

    virtual double eTrainingVector(int i) const { return getQconst().eTrainingVector(i); }

    virtual int covTrainingVector(gentype &resv, gentype &resmu, int i, int j, gentype ***pxyprodi = NULL, gentype ***pxyprodj = NULL, gentype **pxyprodij = NULL) const { return getQconst().covTrainingVector(resv,resmu,i,j,pxyprodi,pxyprodj,pxyprodij); }

    virtual double         &dedgTrainingVector(double         &res, int i) const { return getQconst().dedgTrainingVector(res,i); }
    virtual Vector<double> &dedgTrainingVector(Vector<double> &res, int i) const { return getQconst().dedgTrainingVector(res,i); }
    virtual d_anion        &dedgTrainingVector(d_anion        &res, int i) const { return getQconst().dedgTrainingVector(res,i); }
    virtual gentype        &dedgTrainingVector(gentype        &res, int i) const { return getQconst().dedgTrainingVector(res,i); }

    virtual double dedgAlphaTrainingVector(int i, int j) const { return getQconst().dedgAlphaTrainingVector(i,j); }
    virtual Matrix<double> &dedgAlphaTrainingVector(Matrix<double> &res) const { return getQconst().dedgAlphaTrainingVector(res); }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, int i) const { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, int i) const { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, int i) const { getQconst().deTrainingVectorX(resx,i); return; }

    virtual void dgTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const { getQconst().dgTrainingVectorX(resx,i); return; }
    virtual void dgTrainingVectorX(Vector<double>  &resx, const Vector<int> &i) const { getQconst().dgTrainingVectorX(resx,i); return; }

    virtual void deTrainingVectorX(Vector<gentype> &resx, const Vector<int> &i) const { getQconst().deTrainingVectorX(resx,i); return; }

    virtual int ggTrainingVector(double &resg,         int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(Vector<double> &resg, int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }
    virtual int ggTrainingVector(d_anion &resg,        int i, int retaltg = 0, gentype ***pxyprodi = NULL) const { return getQconst().ggTrainingVector(resg,i,retaltg,pxyprodi); }

    virtual void dgTrainingVector(Vector<gentype>         &res, gentype        &resn, int i) const { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, double         &resn, int i) const { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, Vector<double> &resn, int i) const { getQconst().dgTrainingVector(res,resn,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, d_anion        &resn, int i) const { getQconst().dgTrainingVector(res,resn,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, gentype &resn, int i) const { getQconst().deTrainingVector(res,resn,i); return; }

    virtual void dgTrainingVector(Vector<gentype>         &res, const Vector<int> &i) const { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<double>          &res, const Vector<int> &i) const { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<Vector<double> > &res, const Vector<int> &i) const { getQconst().dgTrainingVector(res,i); return; }
    virtual void dgTrainingVector(Vector<d_anion>         &res, const Vector<int> &i) const { getQconst().dgTrainingVector(res,i); return; }

    virtual void deTrainingVector(Vector<gentype> &res, const Vector<int> &i) const { getQconst().deTrainingVector(res,i); return; }

    virtual void stabProbTrainingVector(double  &res, int i, int p, double pnrm, int rot, double mu, double B) const { getQconst().stabProbTrainingVector(res,i,p,pnrm,rot,mu,B); return; }

    virtual int gg(               gentype &resg, const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(     resg,x        ,xinf,pxyprodx); }
    virtual int hh(gentype &resh,                const SparseVector<gentype> &x                 , const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().hh(resh,     x        ,xinf,pxyprodx); }
    virtual int gh(gentype &resh, gentype &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gh(resh,resg,x,retaltg,xinf,pxyprodx); }

    virtual double e(const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { return getQconst().e(y,x,xinf); }

    virtual int cov(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const SparseVector<gentype> &xb, const vecInfo *xainf = NULL, const vecInfo *xbinf = NULL, gentype ***pxyprodx = NULL, gentype ***pxyprody = NULL, gentype **pxyprodij = NULL) const { return getQconst().cov(resv,resmu,xa,xb,xainf,xbinf,pxyprodx,pxyprody,pxyprodij); }

    virtual void dedg(double         &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(Vector<double> &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(d_anion        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dedg(res,y,x,xinf); return; }
    virtual void dedg(gentype        &res, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dedg(res,y,x,xinf); return; }

    virtual void dgX(Vector<gentype> &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dgX(resx,x,xinf); return; }
    virtual void dgX(Vector<double>  &resx, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dgX(resx,x,xinf); return; }

    virtual void deX(Vector<gentype> &resx, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().deX(resx,y,x,xinf); return; }

    virtual int gg(double &resg,         const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(Vector<double> &resg, const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }
    virtual int gg(d_anion &resg,        const SparseVector<gentype> &x, int retaltg = 0, const vecInfo *xinf = NULL, gentype ***pxyprodx = NULL) const { return getQconst().gg(resg,x,retaltg,xinf,pxyprodx); }

    virtual void dg(Vector<gentype>         &res, gentype        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<double>          &res, double         &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<Vector<double> > &res, Vector<double> &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,resn,x,xinf); return; }
    virtual void dg(Vector<d_anion>         &res, d_anion        &resn, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().dg(res,resn,x,xinf); return; }

    virtual void de(Vector<gentype> &res, gentype &resn, const gentype &y, const SparseVector<gentype> &x, const vecInfo *xinf = NULL) const { getQconst().de(res,resn,y,x,xinf); return; }

    virtual void stabProb(double  &res, const SparseVector<gentype> &x, int p, double pnrm, int rot, double mu, double B) const { getQconst().stabProb(res,x,p,pnrm,rot,mu,B); return; }

    // var and covar functions

    virtual int varTrainingVector(gentype &resv, gentype &resmu, int i, gentype ***pxyprodi = NULL, gentype **pxyprodii = NULL) const { return getQconst().varTrainingVector(resv,resmu,i,pxyprodi,pxyprodii); }
    virtual int var(gentype &resv, gentype &resmu, const SparseVector<gentype> &xa, const vecInfo *xainf = NULL, gentype ***pxyprodx = NULL, gentype **pxyprodxx = NULL) const { return getQconst().var(resv,resmu,xa,xainf,pxyprodx,pxyprodxx); }

    virtual int covarTrainingVector(Matrix<gentype> &resv, const Vector<int> &i) const { return getQconst().covarTrainingVector(resv,i); }
    virtual int covar(Matrix<gentype> &resv, const Vector<SparseVector<gentype> > &x) const { return getQconst().covar(resv,x); }

    // Training data tracking functions:

    virtual const Vector<int>          &indKey(void)          const { return getQconst().indKey();          }
    virtual const Vector<int>          &indKeyCount(void)     const { return getQconst().indKeyCount();     }
    virtual const Vector<int>          &dattypeKey(void)      const { return getQconst().dattypeKey();      }
    virtual const Vector<Vector<int> > &dattypeKeyBreak(void) const { return getQconst().dattypeKeyBreak(); }

    // Other functions

    virtual void setaltx(const ML_Base *_altxsrc) { Qatonce.setaltx(_altxsrc); Qredbin.setaltx(_altxsrc); QMatonce.setaltx(_altxsrc); QMredbin.setaltx(_altxsrc); return; }

    virtual int disable(int i)                { return getQ().disable(i); }
    virtual int disable(const Vector<int> &i) { return getQ().disable(i); }

    // Training data information functions (all assume no far/farfar/farfarfar or multivectors)

    virtual const SparseVector<gentype> &xsum      (SparseVector<gentype> &res) const { return getQconst().xsum(res);       }
    virtual const SparseVector<gentype> &xmean     (SparseVector<gentype> &res) const { return getQconst().xmean(res);      }
    virtual const SparseVector<gentype> &xmeansq   (SparseVector<gentype> &res) const { return getQconst().xmeansq(res);    }
    virtual const SparseVector<gentype> &xsqsum    (SparseVector<gentype> &res) const { return getQconst().xsqsum(res);     }
    virtual const SparseVector<gentype> &xsqmean   (SparseVector<gentype> &res) const { return getQconst().xsqmean(res);    }
    virtual const SparseVector<gentype> &xmedian   (SparseVector<gentype> &res) const { return getQconst().xmedian(res);    }
    virtual const SparseVector<gentype> &xvar      (SparseVector<gentype> &res) const { return getQconst().xvar(res);       }
    virtual const SparseVector<gentype> &xstddev   (SparseVector<gentype> &res) const { return getQconst().xstddev(res);    }
    virtual const SparseVector<gentype> &xmax      (SparseVector<gentype> &res) const { return getQconst().xmax(res);       }
    virtual const SparseVector<gentype> &xmin      (SparseVector<gentype> &res) const { return getQconst().xmin(res);       }

    // Kernel normalisation function

    virtual int normKernelZeroMeanUnitVariance  (int flatnorm = 0, int noshift = 0) { return getQ().normKernelZeroMeanUnitVariance(flatnorm,noshift);   }
    virtual int normKernelZeroMedianUnitVariance(int flatnorm = 0, int noshift = 0) { return getQ().normKernelZeroMedianUnitVariance(flatnorm,noshift); }
    virtual int normKernelUnitRange             (int flatnorm = 0, int noshift = 0) { return getQ().normKernelUnitRange(flatnorm,noshift);              }

    // Helper functions for sparse variables

    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<gentype>      &src) const { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const Vector<double>       &src) const { return getQconst().xlateToSparse(dest,src); }
    virtual SparseVector<gentype> &xlateToSparse(SparseVector<gentype> &dest, const SparseVector<double> &src) const { return getQconst().xlateToSparse(dest,src); }

    virtual Vector<gentype> &xlateFromSparse(Vector<gentype> &dest, const SparseVector<gentype> &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<gentype> &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const SparseVector<double>  &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<gentype>       &src) const { return getQconst().xlateFromSparse(dest,src); }
    virtual Vector<double>  &xlateFromSparse(Vector<double>  &dest, const Vector<double>        &src) const { return getQconst().xlateFromSparse(dest,src); }

    virtual Vector<double>  &xlateFromSparseTrainingVector(Vector<double>  &dest, int i) const { return getQconst().xlateFromSparseTrainingVector(dest,i); }
    virtual Vector<gentype> &xlateFromSparseTrainingVector(Vector<gentype> &dest, int i) const { return getQconst().xlateFromSparseTrainingVector(dest,i); }

    virtual SparseVector<gentype> &makeFullSparse(SparseVector<gentype> &dest) const { return getQconst().makeFullSparse(dest); }

    // x detangling

    virtual int detangle_x(int i, int usextang = 0) const
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
                   int &iplanr, int &iplan, int &iset, int usextang = 1, int allocxuntangifneeded = 1) const
    {
        return getQconst().detangle_x(xuntang,xinfountang,xnear,xfar,xfarfar,xnearinfo,xfarinfo,inear,ifar,ineartup,ifartup,ilr,irr,igr,iokr,iok,i,idiagr,xx,xxinfo,gradOrder,iplanr,iplan,iset,usextang,allocxuntangifneeded);
    }









    // ================================================================
    //     Common functions for all SVMs
    // ================================================================

    virtual       SVM_Generic &getSVM     (void)       { return *this; }
    virtual const SVM_Generic &getSVMconst(void) const { return *this; }

    // Constructors, destructors, assignment etc..

    virtual int setAlpha(const Vector<gentype> &newAlpha) { return getQ().setAlpha(newAlpha); }
    virtual int setBias (const gentype         &newBias ) { return getQ().setBias (newBias);  }

    virtual int setAlphaR(const Vector<double>          &newAlpha) { return getQ().setAlphaR(newAlpha); }
    virtual int setAlphaV(const Vector<Vector<double> > &newAlpha) { return getQ().setAlphaV(newAlpha); }
    virtual int setAlphaA(const Vector<d_anion>         &newAlpha) { return getQ().setAlphaA(newAlpha); }

    virtual int setBiasR(const double         &newBias) { return getQ().setBiasR(newBias); }
    virtual int setBiasV(const Vector<double> &newBias) { return getQ().setBiasV(newBias); }
    virtual int setBiasA(const d_anion        &newBias) { return getQ().setBiasA(newBias); }

    // Information functions (training data):

    virtual int NZ (void)  const { return getQconst().NZ ();  }
    virtual int NF (void)  const { return getQconst().NF ();  }
    virtual int NS (void)  const { return getQconst().NS ();  }
    virtual int NC (void)  const { return getQconst().NC ();  }
    virtual int NLB(void)  const { return getQconst().NLB();  }
    virtual int NLF(void)  const { return getQconst().NLF();  }
    virtual int NUF(void)  const { return getQconst().NUF();  }
    virtual int NUB(void)  const { return getQconst().NUB();  }
    virtual int NF (int q) const { return getQconst().NF (q); }
    virtual int NZ (int q) const { return getQconst().NZ (q); }
    virtual int NS (int q) const { return getQconst().NS (q); }
    virtual int NC (int q) const { return getQconst().NC (q); }
    virtual int NLB(int q) const { return getQconst().NLB(q); }
    virtual int NLF(int q) const { return getQconst().NLF(q); }
    virtual int NUF(int q) const { return getQconst().NUF(q); }
    virtual int NUB(int q) const { return getQconst().NUB(q); }

    virtual const Vector<Vector<int> > &ClassRep(void)  const { return getQconst().ClassRep();    }
    virtual int                         findID(int ref) const;

    virtual int isLinearCost(void)    const { return getQconst().isLinearCost();    }
    virtual int isQuadraticCost(void) const { return getQconst().isQuadraticCost(); }
    virtual int is1NormCost(void)     const { return getQconst().is1NormCost();     }
    virtual int isVarBias(void)       const { return getQconst().isVarBias();       }
    virtual int isPosBias(void)       const { return getQconst().isPosBias();       }
    virtual int isNegBias(void)       const { return getQconst().isNegBias();       }
    virtual int isFixedBias(void)     const { return getQconst().isFixedBias();     }
    virtual int isVarBias(int dq)     const { return getQconst().isVarBias(dq);     }
    virtual int isPosBias(int dq)     const { return getQconst().isPosBias(dq);     }
    virtual int isNegBias(int dq)     const { return getQconst().isNegBias(dq);     }
    virtual int isFixedBias(int dq)   const { return getQconst().isFixedBias(dq);   }

    virtual int isNoMonotonicConstraints(void)    const { return getQconst().isNoMonotonicConstraints();    }
    virtual int isForcedMonotonicIncreasing(void) const { return getQconst().isForcedMonotonicIncreasing(); }
    virtual int isForcedMonotonicDecreasing(void) const { return getQconst().isForcedMonotonicDecreasing(); }

    virtual int isOptActive(void) const { return getQconst().isOptActive(); }
    virtual int isOptSMO(void)    const { return getQconst().isOptSMO();    }
    virtual int isOptD2C(void)    const { return getQconst().isOptD2C();    }
    virtual int isOptGrad(void)   const { return getQconst().isOptGrad();   }

    virtual int isFixedTube(void)  const { return getQconst().isFixedTube(); }
    virtual int isShrinkTube(void) const { return getQconst().isShrinkTube(); }

    virtual int isRestrictEpsPos(void) const { return getQconst().isRestrictEpsPos(); }
    virtual int isRestrictEpsNeg(void) const { return getQconst().isRestrictEpsNeg(); }

    virtual int isClassifyViaSVR(void) const { return getQconst().isClassifyViaSVR(); }
    virtual int isClassifyViaSVM(void) const { return getQconst().isClassifyViaSVM(); }

    virtual int is1vsA(void)    const { return getQconst().is1vsA();    }
    virtual int is1vs1(void)    const { return getQconst().is1vs1();    }
    virtual int isDAGSVM(void)  const { return getQconst().isDAGSVM();  }
    virtual int isMOC(void)     const { return getQconst().isMOC();     }
    virtual int ismaxwins(void) const { return getQconst().ismaxwins(); }
    virtual int isrecdiv(void)  const { return getQconst().isrecdiv();  }

    virtual int isatonce(void) const { return getQconst().isatonce(); }
    virtual int isredbin(void) const { return getQconst().isredbin(); }

    virtual int isKreal(void)   const { return getQconst().isKreal();   }
    virtual int isKunreal(void) const { return getQconst().isKunreal(); }

    virtual int isanomalyOn(void)  const { return getQconst().isanomalyOn();  }
    virtual int isanomalyOff(void) const { return getQconst().isanomalyOff(); }

    virtual int isautosetOff(void)          const { return getQconst().isautosetOff();          }
    virtual int isautosetCscaled(void)      const { return getQconst().isautosetCscaled();      }  
    virtual int isautosetCKmean(void)       const { return getQconst().isautosetCKmean();       }
    virtual int isautosetCKmedian(void)     const { return getQconst().isautosetCKmedian();     }
    virtual int isautosetCNKmean(void)      const { return getQconst().isautosetCNKmean();      }
    virtual int isautosetCNKmedian(void)    const { return getQconst().isautosetCNKmedian();    }
    virtual int isautosetLinBiasForce(void) const { return getQconst().isautosetLinBiasForce(); }

    virtual double outerlr(void)       const { return getQconst().outerlr();       }
    virtual double outermom(void)      const { return getQconst().outermom();      }
    virtual int    outermethod(void)   const { return getQconst().outermethod();   }
    virtual double outertol(void)      const { return getQconst().outertol();      }
    virtual double outerovsc(void)     const { return getQconst().outerovsc();     }
    virtual int    outermaxitcnt(void) const { return getQconst().outermaxitcnt(); }
    virtual int    outermaxcache(void) const { return getQconst().outermaxcache(); }

    virtual       int      maxiterfuzzt(void) const { return getQconst().maxiterfuzzt(); }
    virtual       int      usefuzzt(void)     const { return getQconst().usefuzzt();     }
    virtual       double   lrfuzzt(void)      const { return getQconst().lrfuzzt();      }
    virtual       double   ztfuzzt(void)      const { return getQconst().ztfuzzt();      }
    virtual const gentype &costfnfuzzt(void)  const { return getQconst().costfnfuzzt();  }

    virtual int m(void) const { return getQconst().m(); }

    virtual double LinBiasForce(void)    const { return getQconst().LinBiasForce();    }
    virtual double QuadBiasForce(void)   const { return getQconst().QuadBiasForce();   }
    virtual double LinBiasForce(int dq)  const { return getQconst().LinBiasForce(dq);  }
    virtual double QuadBiasForce(int dq) const { return getQconst().QuadBiasForce(dq); }

    virtual double nu(void)     const { return getQconst().nu();     }
    virtual double nuQuad(void) const { return getQconst().nuQuad(); }

    virtual double anomalyNu(void)    const { return getQconst().anomalyNu();    }
    virtual int    anomalyClass(void) const { return getQconst().anomalyClass(); }

    virtual double autosetCval(void)  const { return getQconst().autosetCval();  }
    virtual double autosetnuval(void) const { return getQconst().autosetnuval(); }

    virtual int anomclass(void) const { return getQconst().anomclass(); }

    virtual const Matrix<double>          &Gp         (void)        const { return getQconst().Gp();          }
    virtual const Matrix<double>          &XX         (void)        const { return getQconst().XX();          }
    virtual const Vector<double>          &kerndiag   (void)        const { return getQconst().kerndiag();    }
    virtual const Vector<Vector<double> > &getu       (void)        const { return getQconst().getu();        }
    virtual const gentype                 &bias       (void)        const { return getQconst().bias();        }
    virtual const Vector<gentype>         &alpha      (void)        const { return getQconst().alpha();       }
    virtual const Vector<double>          &zR         (void)        const { return getQconst().zR();          }
    virtual const Vector<Vector<double> > &zV         (void)        const { return getQconst().zV();          }
    virtual const Vector<d_anion>         &zA         (void)        const { return getQconst().zA();          }
    virtual const double                  &biasR      (void)        const { return getQconst().biasR();       }
    virtual const Vector<double>          &biasV      (int raw = 0) const { return getQconst().biasV(raw);    }
    virtual const d_anion                 &biasA      (void)        const { return getQconst().biasA();       }
    virtual const Vector<double>          &alphaR     (void)        const { return getQconst().alphaR();      }
    virtual const Vector<Vector<double> > &alphaV     (int raw = 0) const { return getQconst().alphaV(raw);   }
    virtual const Vector<d_anion>         &alphaA     (void)        const { return getQconst().alphaA();      }

    virtual const double         &zR(int i) const { return getQconst().zR(i); }
    virtual const Vector<double> &zV(int i) const { return getQconst().zV(i); }
    virtual const d_anion        &zA(int i) const { return getQconst().zA(i); }

    // Training set modification:

    virtual int removeNonSupports(void)      { return getQ().removeNonSupports();      }
    virtual int trimTrainingSet(int maxsize) { return getQ().trimTrainingSet(maxsize); }

    // General modification and autoset functions

    virtual int setLinearCost(void)                        { int res = Qatonce.setLinearCost();    res |= Qredbin.setLinearCost();    res |= QMatonce.setLinearCost();    res |= QMredbin.setLinearCost();    return res; }
    virtual int setQuadraticCost(void)                     { int res = Qatonce.setQuadraticCost(); res |= Qredbin.setQuadraticCost(); res |= QMatonce.setQuadraticCost(); res |= QMredbin.setQuadraticCost(); return res; }
    virtual int set1NormCost(void)                         { int res = Qatonce.set1NormCost();     res |= Qredbin.set1NormCost();     res |= QMatonce.set1NormCost();     res |= QMredbin.set1NormCost();     return res; }
    virtual int setVarBias(void)                           { int res = Qatonce.setVarBias();            res |= Qredbin.setVarBias();            res |= QMatonce.setVarBias();            res |= QMredbin.setVarBias();            return res; }
    virtual int setPosBias(void)                           { int res = Qatonce.setPosBias();            res |= Qredbin.setPosBias();            res |= QMatonce.setPosBias();            res |= QMredbin.setPosBias();            return res; }
    virtual int setNegBias(void)                           { int res = Qatonce.setNegBias();            res |= Qredbin.setNegBias();            res |= QMatonce.setNegBias();            res |= QMredbin.setNegBias();            return res; }
    virtual int setFixedBias(double newbias = 0.0)         { return getQ().setFixedBias(newbias); }
    virtual int setVarBias(int  q)                         { int res = Qatonce.setVarBias(q);           res |= Qredbin.setVarBias(q);           res |= QMatonce.setVarBias(q);           res |= QMredbin.setVarBias(q);           return res; }
    virtual int setPosBias(int  q)                         { int res = Qatonce.setPosBias(q);           res |= Qredbin.setPosBias(q);           res |= QMatonce.setPosBias(q);           res |= QMredbin.setPosBias(q);           return res; }
    virtual int setNegBias(int  q)                         { int res = Qatonce.setNegBias(q);           res |= Qredbin.setNegBias(q);           res |= QMatonce.setNegBias(q);           res |= QMredbin.setNegBias(q);           return res; }
    virtual int setFixedBias(int  q, double newbias = 0.0) { int res = Qatonce.setFixedBias(q,newbias); res |= Qredbin.setFixedBias(q,newbias); res |= QMatonce.setFixedBias(q,newbias); res |= QMredbin.setFixedBias(q,newbias); return res; }
    virtual int setFixedBias(const gentype &newbias)       { int res = Qatonce.setFixedBias(newbias);   res |= Qredbin.setFixedBias(newbias);   res |= QMatonce.setFixedBias(newbias);   res |= QMredbin.setFixedBias(newbias);   return res; }

    virtual int setNoMonotonicConstraints(void)    { return getQ().setNoMonotonicConstraints();    }
    virtual int setForcedMonotonicIncreasing(void) { return getQ().setForcedMonotonicIncreasing(); }
    virtual int setForcedMonotonicDecreasing(void) { return getQ().setForcedMonotonicDecreasing(); }

    virtual int setOptActive(void) { int res = Qatonce.setOptActive(); res |= Qredbin.setOptActive(); res |= QMatonce.setOptActive(); res |= QMredbin.setOptActive(); return res; }
    virtual int setOptSMO(void)    { int res = Qatonce.setOptSMO();    res |= Qredbin.setOptSMO();    res |= QMatonce.setOptSMO();    res |= QMredbin.setOptSMO();    return res; }
    virtual int setOptD2C(void)    { int res = Qatonce.setOptD2C();    res |= Qredbin.setOptD2C();    res |= QMatonce.setOptD2C();    res |= QMredbin.setOptD2C();    return res; }
    virtual int setOptGrad(void)   { int res = Qatonce.setOptGrad();   res |= Qredbin.setOptGrad();   res |= QMatonce.setOptGrad();   res |= QMredbin.setOptGrad();   return res; }

    virtual int setFixedTube(void)  { int res = Qatonce.setFixedTube();  res |= Qredbin.setFixedTube();  res |= QMatonce.setFixedTube();  res |= QMredbin.setFixedTube();  return res; }
    virtual int setShrinkTube(void) { int res = Qatonce.setShrinkTube(); res |= Qredbin.setShrinkTube(); res |= QMatonce.setShrinkTube(); res |= QMredbin.setShrinkTube(); return res; }

    virtual int setRestrictEpsPos(void) { int res = Qatonce.setRestrictEpsPos(); res |= Qredbin.setRestrictEpsPos(); res |= QMatonce.setRestrictEpsPos(); res |= QMredbin.setRestrictEpsPos(); return res; }
    virtual int setRestrictEpsNeg(void) { int res = Qatonce.setRestrictEpsNeg(); res |= Qredbin.setRestrictEpsNeg(); res |= QMatonce.setRestrictEpsNeg(); res |= QMredbin.setRestrictEpsNeg(); return res; }

    virtual int setClassifyViaSVR(void) { int res = Qatonce.setClassifyViaSVR(); res |= Qredbin.setClassifyViaSVR(); res |= QMatonce.setClassifyViaSVR(); res |= QMredbin.setClassifyViaSVR(); return res; }
    virtual int setClassifyViaSVM(void) { int res = Qatonce.setClassifyViaSVM(); res |= Qredbin.setClassifyViaSVM(); res |= QMatonce.setClassifyViaSVM(); res |= QMredbin.setClassifyViaSVM(); return res; }

    virtual int set1vsA(void)    { int res = Qatonce.set1vsA();    res |= Qredbin.set1vsA();    res |= QMatonce.set1vsA();    res |= QMredbin.set1vsA();    return res; }
    virtual int set1vs1(void)    { int res = Qatonce.set1vs1();    res |= Qredbin.set1vs1();    res |= QMatonce.set1vs1();    res |= QMredbin.set1vs1();    return res; }
    virtual int setDAGSVM(void)  { int res = Qatonce.setDAGSVM();  res |= Qredbin.setDAGSVM();  res |= QMatonce.setDAGSVM();  res |= QMredbin.setDAGSVM();  return res; }
    virtual int setMOC(void)     { int res = Qatonce.setMOC();     res |= Qredbin.setMOC();     res |= QMatonce.setMOC();     res |= QMredbin.setMOC();     return res; }
    virtual int setmaxwins(void) { int res = Qatonce.setmaxwins(); res |= Qredbin.setmaxwins(); res |= QMatonce.setmaxwins(); res |= QMredbin.setmaxwins(); return res; }
    virtual int setrecdiv(void)  { int res = Qatonce.setrecdiv();  res |= Qredbin.setrecdiv();  res |= QMatonce.setrecdiv();  res |= QMredbin.setrecdiv();  return res; }

    virtual int setatonce(void);
    virtual int setredbin(void);

    virtual int setKreal(void);
    virtual int setKunreal(void);

    virtual int anomalyOn(int danomalyClass, double danomalyNu) { int res = Qatonce.anomalyOn(danomalyClass,danomalyNu); res |= Qredbin.anomalyOn(danomalyClass,danomalyNu); res |= QMatonce.anomalyOn(danomalyClass,danomalyNu); res |= QMredbin.anomalyOn(danomalyClass,danomalyNu); return res; }
    virtual int anomalyOff(void)                                { int res = Qatonce.anomalyOff();                        res |= Qredbin.anomalyOff();                        res |= QMatonce.anomalyOff();                        res |= QMredbin.anomalyOff();                        return res; }

    virtual int setouterlr(double xouterlr)           { int res = Qatonce.setouterlr(xouterlr);              res |= Qredbin.setouterlr(xouterlr);              res |= QMatonce.setouterlr(xouterlr);              res |= QMredbin.setouterlr(xouterlr);              return res; }
    virtual int setoutermom(double xoutermom)         { int res = Qatonce.setoutermom(xoutermom);            res |= Qredbin.setoutermom(xoutermom);            res |= QMatonce.setoutermom(xoutermom);            res |= QMredbin.setoutermom(xoutermom);            return res; }
    virtual int setoutermethod(int xoutermethod)      { int res = Qatonce.setoutermethod(xoutermethod);      res |= Qredbin.setoutermethod(xoutermethod);      res |= QMatonce.setoutermethod(xoutermethod);      res |= QMredbin.setoutermethod(xoutermethod);      return res; }
    virtual int setoutertol(double xoutertol)         { int res = Qatonce.setoutertol(xoutertol);            res |= Qredbin.setoutertol(xoutertol);            res |= QMatonce.setoutertol(xoutertol);            res |= QMredbin.setoutertol(xoutertol);            return res; }
    virtual int setouterovsc(double xouterovsc)       { int res = Qatonce.setouterovsc(xouterovsc);          res |= Qredbin.setouterovsc(xouterovsc);          res |= QMatonce.setouterovsc(xouterovsc);          res |= QMredbin.setouterovsc(xouterovsc);          return res; }
    virtual int setoutermaxitcnt(int xoutermaxits)    { int res = Qatonce.setoutermaxitcnt(xoutermaxits);    res |= Qredbin.setoutermaxitcnt(xoutermaxits);    res |= QMatonce.setoutermaxitcnt(xoutermaxits);    res |= QMredbin.setoutermaxitcnt(xoutermaxits);    return res; }
    virtual int setoutermaxcache(int xoutermaxcacheN) { int res = Qatonce.setoutermaxcache(xoutermaxcacheN); res |= Qredbin.setoutermaxcache(xoutermaxcacheN); res |= QMatonce.setoutermaxcache(xoutermaxcacheN); res |= QMredbin.setoutermaxcache(xoutermaxcacheN); return res; }

    virtual int setmaxiterfuzzt(int xmaxiterfuzzt)              { int res = Qatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= Qredbin.setmaxiterfuzzt(xmaxiterfuzzt); res |= QMatonce.setmaxiterfuzzt(xmaxiterfuzzt); res |= QMredbin.setmaxiterfuzzt(xmaxiterfuzzt); return res; }
    virtual int setusefuzzt(int xusefuzzt)                      { int res = Qatonce.setusefuzzt(xusefuzzt);         res |= Qredbin.setusefuzzt(xusefuzzt);         res |= QMatonce.setusefuzzt(xusefuzzt);         res |= QMredbin.setusefuzzt(xusefuzzt);         return res; }
    virtual int setlrfuzzt(double xlrfuzzt)                     { int res = Qatonce.setlrfuzzt(xlrfuzzt);           res |= Qredbin.setlrfuzzt(xlrfuzzt);           res |= QMatonce.setlrfuzzt(xlrfuzzt);           res |= QMredbin.setlrfuzzt(xlrfuzzt);           return res; }
    virtual int setztfuzzt(double xztfuzzt)                     { int res = Qatonce.setztfuzzt(xztfuzzt);           res |= Qredbin.setztfuzzt(xztfuzzt);           res |= QMatonce.setztfuzzt(xztfuzzt);           res |= QMredbin.setztfuzzt(xztfuzzt);           return res; }
    virtual int setcostfnfuzzt(const gentype &xcostfnfuzzt)     { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   res |= QMatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= QMredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }
    virtual int setcostfnfuzzt(const std::string &xcostfnfuzzt) { int res = Qatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= Qredbin.setcostfnfuzzt(xcostfnfuzzt);   res |= QMatonce.setcostfnfuzzt(xcostfnfuzzt);   res |= QMredbin.setcostfnfuzzt(xcostfnfuzzt);   return res; }

    virtual int setm(int xm) { int res = Qatonce.setm(xm); res |= Qredbin.setm(xm); res |= QMatonce.setm(xm); res |= QMredbin.setm(xm); return res; }

    virtual int setLinBiasForce(double newval)          { int res = Qatonce.setLinBiasForce(newval);    res |= Qredbin.setLinBiasForce(newval);    res |= QMatonce.setLinBiasForce(newval);    res |= QMredbin.setLinBiasForce(newval);    return res; }
    virtual int setQuadBiasForce(double newval)         { int res = Qatonce.setQuadBiasForce(newval);   res |= Qredbin.setQuadBiasForce(newval);   res |= QMatonce.setQuadBiasForce(newval);   res |= QMredbin.setQuadBiasForce(newval);   return res; }
    virtual int setLinBiasForce(int  q, double newval)  { int res = Qatonce.setLinBiasForce(q,newval);  res |= Qredbin.setLinBiasForce(q,newval);  res |= QMatonce.setLinBiasForce(q,newval);  res |= QMredbin.setLinBiasForce(q,newval);  return res; }
    virtual int setQuadBiasForce(int  q, double newval) { int res = Qatonce.setQuadBiasForce(q,newval); res |= Qredbin.setQuadBiasForce(q,newval); res |= QMatonce.setQuadBiasForce(q,newval); res |= QMredbin.setQuadBiasForce(q,newval); return res; }

    virtual int setnu(double xnu)         { int res = Qatonce.setnu(xnu);         res |= Qredbin.setnu(xnu);         res |= QMatonce.setnu(xnu);         res |= QMredbin.setnu(xnu);         return res; }
    virtual int setnuQuad(double xnuQuad) { int res = Qatonce.setnuQuad(xnuQuad); res |= Qredbin.setnuQuad(xnuQuad); res |= QMatonce.setnuQuad(xnuQuad); res |= QMredbin.setnuQuad(xnuQuad); return res; }

    virtual int autosetOff(void)                                     { int res = Qatonce.autosetOff();         res |= Qredbin.autosetOff();         res |= QMatonce.autosetOff();                    res |= QMredbin.autosetOff();                    return res; }
    virtual int autosetCscaled(double Cval)                          { int res = Qatonce.autosetCscaled(Cval); res |= Qredbin.autosetCscaled(Cval); res |= QMatonce.autosetCscaled(Cval);            res |= QMredbin.autosetCscaled(Cval);            return res; }
    virtual int autosetCKmean(void)                                  { int res = Qatonce.autosetCKmean();      res |= Qredbin.autosetCKmean();      res |= QMatonce.autosetCKmean();                 res |= QMredbin.autosetCKmean();                 return res; }
    virtual int autosetCKmedian(void)                                { int res = Qatonce.autosetCKmedian();    res |= Qredbin.autosetCKmedian();    res |= QMatonce.autosetCKmedian();               res |= QMredbin.autosetCKmedian();               return res; }
    virtual int autosetCNKmean(void)                                 { int res = Qatonce.autosetCNKmean();     res |= Qredbin.autosetCNKmean();     res |= QMatonce.autosetCNKmean();                res |= QMredbin.autosetCNKmean();                return res; }
    virtual int autosetCNKmedian(void)                               { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   res |= QMatonce.autosetCNKmedian();              res |= QMredbin.autosetCNKmedian();              return res; }
    virtual int autosetLinBiasForce(double nuval, double Cval = 1.0) { int res = Qatonce.autosetCNKmedian();   res |= Qredbin.autosetCNKmedian();   res |= QMatonce.autosetLinBiasForce(nuval,Cval); res |= QMredbin.autosetLinBiasForce(nuval,Cval); return res; }

    virtual void setanomalyclass(int n) { getQ().setanomalyclass(n); return; }

    // Evaluation Functions:

    virtual double quasiloglikelihood(void) const { return getQconst().quasiloglikelihood(); }

protected:
    // ================================================================
    //     Base level functions
    // ================================================================

    // SVM specific

    virtual int addTrainingVector (int i, const Vector<double> &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);
    virtual int qaddTrainingVector(int i, const Vector<double> &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1, int d = 2);

    virtual int addTrainingVector (int i, const Vector<Vector<double> > &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);
    virtual int qaddTrainingVector(int i, const Vector<Vector<double> > &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh, const Vector<int> &d);

private:

    virtual       SVM_Generic &getQ     (void)       { if ( isQatonce &&  isQreal ) { return static_cast<      SVM_Generic &>(Qatonce);  } else if ( isQatonce && !isQreal ) { return static_cast<      SVM_Generic &>(QMatonce); } else if ( !isQatonce && isQreal ) { return static_cast<      SVM_Generic &>(Qredbin); } return static_cast<      SVM_Generic &>(QMredbin); }
    virtual const SVM_Generic &getQconst(void) const { if ( isQatonce &&  isQreal ) { return static_cast<const SVM_Generic &>(Qatonce);  } else if ( isQatonce && !isQreal ) { return static_cast<const SVM_Generic &>(QMatonce); } else if ( !isQatonce && isQreal ) { return static_cast<const SVM_Generic &>(Qredbin); } return static_cast<const SVM_Generic &>(QMredbin); }

    int isQatonce;
    int isQreal;

    SVM_Vector_atonce Qatonce;
    SVM_Vector_redbin<SVM_Scalar> Qredbin;

    SVM_Vector_Matonce QMatonce;
    SVM_Vector_Mredbin QMredbin;
};

inline void qswap(SVM_Vector &a, SVM_Vector &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Vector::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Vector &b = dynamic_cast<SVM_Vector &>(bb.getML());

    SVM_Generic::qswapinternal(b);

    qswap(isQatonce,b.isQatonce);
    qswap(isQreal  ,b.isQreal  );
    qswap(Qatonce  ,b.Qatonce  );
    qswap(Qredbin  ,b.Qredbin  );
    qswap(QMatonce ,b.QMatonce );
    qswap(QMredbin ,b.QMredbin );

    return;
}

inline void SVM_Vector::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Vector &b = dynamic_cast<const SVM_Vector &>(bb.getMLconst());

    SVM_Generic::semicopy(b);

         if (  isQatonce &&  isQreal ) { Qatonce. semicopy(b.Qatonce ); }
    else if (  isQatonce && !isQreal ) { QMatonce.semicopy(b.QMatonce); }
    else if ( !isQatonce &&  isQreal ) { Qredbin. semicopy(b.Qredbin ); }
    else                               { QMredbin.semicopy(b.QMredbin); }

    return;
}

inline void SVM_Vector::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Vector &src = dynamic_cast<const SVM_Vector &>(bb.getMLconst());

    SVM_Generic::assign(src,onlySemiCopy);

    isQatonce = src.isQatonce;
    isQreal   = src.isQreal;

    Qatonce.assign(src.Qatonce,onlySemiCopy);
    Qredbin.assign(src.Qredbin,onlySemiCopy);
    QMatonce.assign(src.QMatonce,onlySemiCopy);
    QMredbin.assign(src.QMredbin,onlySemiCopy);

    return;
}

#endif
