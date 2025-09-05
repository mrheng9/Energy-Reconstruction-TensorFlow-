 
namespace ana {
  const ana::Cut kNDRockFilter([](const caf::SRProxy * sr)
			       {
				 if(sr->vtx.nelastic == 0) return false;
				 return 
				   sr->vtx.elastic[0].vtx.x > -180 &&
				   sr->vtx.elastic[0].vtx.x <  180 &&
				   sr->vtx.elastic[0].vtx.y > -180 &&
				   sr->vtx.elastic[0].vtx.y <  180 &&
				   sr->vtx.elastic[0].vtx.z >  20;
			       });


  bool IsGoodProng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.elastic[0].fuzzyk.png[prong_idx].len >= 500) return false;
    if(sr->vtx.elastic[0].fuzzyk.png[prong_idx].cvnpart.muonid <= 0) return false;
    return true;
  }

  bool IsPionPng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;
    if(IsGoodProng(prong_idx, sr))
      return abs(sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg) == 211;
    return false;
  }

  bool IsPhotonPng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;
    if(IsGoodProng(prong_idx, sr))
      return sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg == 22;
      //return abs(sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg) == 22;
    return false;
  }

  bool IsProtonPng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;
    if(IsGoodProng(prong_idx, sr))
      return abs(sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg) == 2212;
    return false;
  }

  bool IsElectronPng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;
    if(IsGoodProng(prong_idx, sr))
      return abs(sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg) == 11;
    return false;
  }

  bool IsMuonPng(unsigned int prong_idx, const caf::SRProxy * sr) {
    if(sr->vtx.nelastic == 0) return false;
    if(sr->vtx.elastic[0].fuzzyk.npng == 0) return false;
    if(IsGoodProng(prong_idx, sr))
      return abs(sr->vtx.elastic[0].fuzzyk.png[prong_idx].truth.pdg) == 13;
    return false;
  }

  double emid(int vtx_idx, int png_idx,const caf::SRProxy * sr) {
    return sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid +
      sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid;
  }
  ///////////////////////////////////////////////////////////////////////////
  //  EMID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kEMPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> emid_scores;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) emid_scores.push_back(emid(vtx_idx, png_idx, sr));
					   }
					 }
					 return emid_scores;
				       });
  const ana::MultiVar kEMPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> emid_scores;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) emid_scores.push_back(emid(vtx_idx, png_idx, sr));
					   }
					 }
					 return emid_scores;
				       });
  const ana::MultiVar kEMPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> emid_scores;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) emid_scores.push_back(emid(vtx_idx, png_idx, sr));
					   }
					 }
					 return emid_scores;
				       });
  const ana::MultiVar kEMPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> emid_scores;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) emid_scores.push_back(emid(vtx_idx, png_idx, sr));
					   }
					 }
					 return emid_scores;
				       });
  const ana::MultiVar kEMPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> emid_scores;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) emid_scores.push_back(emid(vtx_idx, png_idx, sr));
					   }
					 }
					 return emid_scores;
				       });


  ///////////////////////////////////////////////////////////////////////////
  //  PIONID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kPionPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> pionid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) pionid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.pionid);
					   }
					 }
					 return pionid;
				       });
  const ana::MultiVar kPionPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> pionid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) pionid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.pionid);
					   }
					 }
					 return pionid;
				       });
  const ana::MultiVar kPionPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> pionid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) pionid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.pionid);
					   }
					 }
					 return pionid;
				       });
  const ana::MultiVar kPionPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> pionid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) pionid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.pionid);
					   }
					 }
					 return pionid;
				       });
  const ana::MultiVar kPionPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> pionid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) pionid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.pionid);
					   }
					 }
					 return pionid;
				       });

  ///////////////////////////////////////////////////////////////////////////
  //  ELECTRONID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kElectronPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> electronid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) electronid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid);
					   }
					 }
					 return electronid;
				       });
  const ana::MultiVar kElectronPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> electronid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) electronid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid);
					   }
					 }
					 return electronid;
				       });
  const ana::MultiVar kElectronPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> electronid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) electronid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid);
					   }
					 }
					 return electronid;
				       });
  const ana::MultiVar kElectronPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> electronid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) electronid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid);
					   }
					 }
					 return electronid;
				       });
  const ana::MultiVar kElectronPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> electronid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) electronid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.electronid);
					   }
					 }
					 return electronid;
				       });

  ///////////////////////////////////////////////////////////////////////////
  //  MUONID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kMuonPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> muonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) muonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.muonid);
					   }
					 }
					 return muonid;
				       });
  const ana::MultiVar kMuonPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> muonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) muonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.muonid);
					   }
					 }
					 return muonid;
				       });
  const ana::MultiVar kMuonPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> muonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) muonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.muonid);
					   }
					 }
					 return muonid;
				       });
  const ana::MultiVar kMuonPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> muonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) muonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.muonid);
					   }
					 }
					 return muonid;
				       });
  const ana::MultiVar kMuonPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> muonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) muonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.muonid);
					   }
					 }
					 return muonid;
				       });

  ///////////////////////////////////////////////////////////////////////////
  //  PROTONID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kProtonPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> protonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) protonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.protonid);
					   }
					 }
					 return protonid;
				       });
  const ana::MultiVar kProtonPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> protonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) protonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.protonid);
					   }
					 }
					 return protonid;
				       });
  const ana::MultiVar kProtonPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> protonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) protonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.protonid);
					   }
					 }
					 return protonid;
				       });
  const ana::MultiVar kProtonPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> protonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) protonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.protonid);
					   }
					 }
					 return protonid;
				       });
  const ana::MultiVar kProtonPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> protonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) protonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.protonid);
					   }
					 }
					 return protonid;
				       });

  ///////////////////////////////////////////////////////////////////////////
  //  PHOTONID
  ///////////////////////////////////////////////////////////////////////////
  const ana::MultiVar kPhotonPIDTruePion([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> photonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPionPng(png_idx, sr)) photonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid);
					   }
					 }
					 return photonid;
				       });
  const ana::MultiVar kPhotonPIDTrueElectron([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> photonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsElectronPng(png_idx, sr)) photonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid);
					   }
					 }
					 return photonid;
				       });
  const ana::MultiVar kPhotonPIDTrueMuon([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> photonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsMuonPng(png_idx, sr)) photonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid);
					   }
					 }
					 return photonid;
				       });
  const ana::MultiVar kPhotonPIDTruePhoton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> photonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsPhotonPng(png_idx, sr)) photonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid);
					   }
					 }
					 return photonid;
				       });
  const ana::MultiVar kPhotonPIDTrueProton([](const caf::SRProxy * sr)
				       {			   
					 std::vector<double> photonid;
					 for(unsigned int vtx_idx = 0; vtx_idx < sr->vtx.elastic.size(); vtx_idx++) {
					   for(unsigned int png_idx = 0; png_idx < sr->vtx.elastic[vtx_idx].fuzzyk.png.size(); png_idx++) {
					     if(IsProtonPng(png_idx, sr)) photonid.push_back(sr->vtx.elastic[vtx_idx].fuzzyk.png[png_idx].cvnpart.photonid);
					   }
					 }
					 return photonid;
				       });

}
