#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "CAFAna/Core/Cut.h"
#include "StandardRecord/Proxy/SRProxy.h"

#include "NuXAna/Cuts/NusCuts.h"      
#include "NuXAna/Cuts/NusCuts18.h" 
#include "CAFAna/Cuts/NumuCuts.h"      // kNumuDecafPresel
#include "CAFAna/Cuts/NumuCuts2017.h" // kNumuOptimizedContainFD2017, kNumuContainND2017
#include "CAFAna/Cuts/NueCuts2018.h"       // kNue2017ProngContainment, kNue2018Presel
#include "CAFAna/Cuts/Cuts.h"          // kIsFarDet
#include "CAFAna/Cuts/AnalysisMasks.h" // kApplySecondAnalysisMask
#include "NuXAna/Cuts/NusCuts.h"

#include "Prod4ProngCVNScores.h"

#include "TFile.h"

using namespace ana;

////////////////////////////// Slice Quality ////////////////////////////////////
const ana::Cut kHasVtx([](const caf::SRProxy * sr)
		       {
			 return sr->vtx.nelastic > 0;
		       });
const ana::Cut kHasPng([](const caf::SRProxy * sr)
		       {
			 return sr->vtx.elastic[0].fuzzyk.npng > 0;
		       });
const ana::Cut kSliceQuality([](const caf::SRProxy * sr)
			     {
			       if(kIsFarDet(sr))
				 return kHasVtx(sr) && kHasPng(sr); 
			       else
				 return kHasVtx(sr) && kHasPng(sr) && kNDRockFilter(sr);
			     });

////////////////////////////// Numu Cuts ////////////////////////////////////////
const ana::Cut kNumuPreselMinusCCE([](const caf::SRProxy * sr)
				   {
				     if(kIsFarDet(sr))
				       return kNumuBasicQuality(sr) && kNumuOptimizedContainFD2017(sr);
				     else
				       return kNumuBasicQuality(sr) && kNumuContainND2017(sr);
				   });
const ana::Cut kNumuContain([](const caf::SRProxy * sr)
			    {
			      if(kIsFarDet(sr))
				return kNumuOptimizedContainFD2017(sr);
			      else
				return kNumuContainND2017(sr);
			    });

////////////////////////////// Nue Cuts /////////////////////////////////////////
const ana::Var kHitsPerPlane([](const caf::SRProxy * sr)
			     {
			       return sr->sel.nuecosrej.hitsperplane;
			     });
const ana::Cut kNueDQ = (kHitsPerPlane < 8) && kHasVtx && kHasPng;
const ana::Cut kNueBasicPart = kIsFarDet && kNueDQ && kVeto && kApplySecondAnalysisMask;
const ana::Cut kNueContainFD = kNueBasicPart && kNue2017ProngContainment;
const ana::Cut kNueContain([](const caf::SRProxy * sr)
			   {
			     if(kIsFarDet(sr))
			       return kNueContainFD(sr);
			     else
			       return kNue2017NDContain(sr);
			   });
const ana::Cut kNuePresel([](const caf::SRProxy * sr)
			  {
			    if(kIsFarDet(sr))
			      return kNue2018CorePresel(sr);
			    else
			      return kNue2018NDPresel(sr);
			  });

////////////////////////////// Nus Cuts /////////////////////////////////////////
const ana::Cut kNusContain([](const caf::SRProxy * sr)
			   {
			     if(kIsFarDet(sr))
			       return kNus18FDContain(sr);
			     else
			       return kNus18NDContain(sr);
			   });
const ana::Cut kNusPresel([](const caf::SRProxy * sr)
			  {
			    if(kIsFarDet(sr))
			      return kNus18FDPresel(sr);
			    else
			      return kNus18NDPresel(sr);
			  });

////////////////////////////// ORd Cuts /////////////////////////////////////////
const ana::Cut kOrContainment = kNumuContain || kNusContain || kNueContain;
const ana::Cut kOrPreselection = kNuePresel || kNumuPreselMinusCCE || kNusPresel;



void prod4_pid()
{
  std::map<std::string, const MultiVar &> prong_vars;
  std::map<std::string, const Cut &>      selection_cuts;
  std::map<std::string, SpectrumLoader *> loaders;
  /// pid vars
  // muonid
  prong_vars.insert({"muonid_true_muon",     kMuonPIDTrueMuon});
  prong_vars.insert({"muonid_true_electron", kMuonPIDTrueElectron});
  prong_vars.insert({"muonid_true_pion",     kMuonPIDTruePion});
  prong_vars.insert({"muonid_true_proton",   kMuonPIDTrueProton});
  prong_vars.insert({"muonid_true_photon",   kMuonPIDTruePhoton});

  // electronid
  prong_vars.insert({"electronid_true_muon",     kElectronPIDTrueMuon});
  prong_vars.insert({"electronid_true_electron", kElectronPIDTrueElectron});
  prong_vars.insert({"electronid_true_pion",     kElectronPIDTruePion});
  prong_vars.insert({"electronid_true_proton",   kElectronPIDTrueProton});
  prong_vars.insert({"electronid_true_photon",   kElectronPIDTruePhoton});

  // pionid
  prong_vars.insert({"pionid_true_muon",     kPionPIDTrueMuon});
  prong_vars.insert({"pionid_true_electron", kPionPIDTrueElectron});
  prong_vars.insert({"pionid_true_pion",     kPionPIDTruePion});
  prong_vars.insert({"pionid_true_proton",   kPionPIDTrueProton});
  prong_vars.insert({"pionid_true_photon",   kPionPIDTruePhoton});

  // protonid
  prong_vars.insert({"protonid_true_muon",     kProtonPIDTrueMuon});
  prong_vars.insert({"protonid_true_electron", kProtonPIDTrueElectron});
  prong_vars.insert({"protonid_true_pion",     kProtonPIDTruePion});
  prong_vars.insert({"protonid_true_proton",   kProtonPIDTrueProton});
  prong_vars.insert({"protonid_true_photon",   kProtonPIDTruePhoton});

  // photonid
  prong_vars.insert({"photonid_true_muon",     kPhotonPIDTrueMuon});
  prong_vars.insert({"photonid_true_electron", kPhotonPIDTrueElectron});
  prong_vars.insert({"photonid_true_pion",     kPhotonPIDTruePion});
  prong_vars.insert({"photonid_true_proton",   kPhotonPIDTrueProton});
  prong_vars.insert({"photonid_true_photon",   kPhotonPIDTruePhoton});

  // photonid
  prong_vars.insert({"emid_true_muon",     kEMPIDTrueMuon});
  prong_vars.insert({"emid_true_electron", kEMPIDTrueElectron});
  prong_vars.insert({"emid_true_pion",     kEMPIDTruePion});
  prong_vars.insert({"emid_true_proton",   kEMPIDTrueProton});
  prong_vars.insert({"emid_true_photon",   kEMPIDTruePhoton});

  /// selection cuts
  /// CutVarCache doesn't like this....
  // selection_cuts.insert({"Veto",         Cut(kVeto)});
  // selection_cuts.insert({"Containment",  Cut(kVeto && kOrContainment)});
  // selection_cuts.insert({"Preselection", Cut(kVeto && kOrContainment && kOrPreselection)});
  // dont know why this has to be so complicated
  std::vector<const Cut *> cuts = {new Cut(kVeto), 
				   new Cut(kVeto && kOrContainment),
				   new Cut(kVeto && kOrContainment && kOrPreselection)};
  std::vector<std::string> cut_labels = {"Veto", "Containment", "Preselection"};


  /// loaders
  loaders.insert({"FD_Fluxswap_FHC", new SpectrumLoader("prod_caf_R17-11-14-prod4reco.d_fd_genie_fluxswap_fhc_nova_v08_full_v1")});
  loaders.insert({"FD_Nonswap_FHC",  new SpectrumLoader("prod_caf_R17-11-14-prod4reco.d_fd_genie_nonswap_fhc_nova_v08_full_v1")});
  // loaders.insert({"ND_Nonswap_RHC",  new SpectrumLoader("prod_caf_R17-11-14-prod4reco.CVNprong-respin.a_nd_genie_nonswap_rhc_nova_v08_full_v1")});
  // loaders.insert({"ND_Nonswap_FHC",  new SpectrumLoader("prod_caf_R17-11-14-prod4reco.CVNprong-respin.a_nd_genie_nonswap_fhc_nova_v08_full_v1")});
  loaders.insert({"ND_Nonswap_RHC",  new SpectrumLoader("prod_caf_R17-11-14-prod4reco.neutron-respin.b_nd_genie_nonswap_rhc_nova_v08_full_v1")});
  loaders.insert({"ND_Nonswap_FHC",  new SpectrumLoader("prod_caf_R17-11-14-prod4reco.neutron-respin.b_nd_genie_nonswap_fhc_nova_v08_full_v1")});


  const Binning pid_bins = Binning::Simple(30, 0, 1);

  std::cout << "Prong Vars: " << prong_vars.size() << std::endl;
  std::cout << "Selection Cuts: " << cuts.size() << std::endl;
  std::cout << "Loaders: " << loaders.size() << std::endl;
  
  std::vector<Spectrum*> spectra;
  int start_cut = 1;
  int end_cut = 2;
  for(std::map<std::string, const MultiVar &>::iterator var = prong_vars.begin(); var != prong_vars.end();var++) {
    for(int icut = 0; icut < (int) cuts.size(); icut++) {
      //for(int icut = start_cut; icut < end_cut; icut++) {
      for(std::map<std::string, SpectrumLoader*>::iterator loader = loaders.begin(); loader != loaders.end(); loader++) {
	spectra.push_back(new Spectrum(cut_labels[icut] + "_" + loader->first + "_" + var->first,
				       pid_bins,
				       *loader->second,
				       var->second,
				       kSliceQuality && *cuts[icut]));
      }
    }
  }

  for(std::map<std::string, SpectrumLoader*>::iterator loader = loaders.begin(); loader != loaders.end(); loader++) {
    loader->second->Go();
  }

  TFile * output = new TFile("prod4_pid.root", "recreate");
  for(int i = 0; i < (int) spectra.size(); i++) {
    spectra[i]->SaveTo(output->mkdir(spectra[i]->GetLabels()[0].c_str()));
  }
					
}
