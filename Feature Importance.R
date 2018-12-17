## Calling the package

library(Boruta)

## Reading the data     

train_df <- read.csv("undersample/train_undersample.csv", header = T, sep=',', stringsAsFactors = FALSE)


boruta.train <- Boruta( cible ~ activite_nce + anciennete_client + canal_de_vente + couleur_tarif_elec + duree + echeance_mois + entite_societe_contractante + libelle_naf + marche_de_la_sc + nb_dem_12 + nb_dem_reco_12 + nb_recla_12 + nb_recla_reco_12 + orientation_economique + prix_elec_m3 + prix_gaz_m3 + produit + profil_prm + segment_societe_contractante + type_client + type_d_offre + type_de_prix + volume_annuel + zone, data = train_df, doTrace = 2)

print(boruta.train)


## Plotting

plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
  boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1, las = 2, labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory),cex.axis = 0.7)

## Final Boruta

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

## Obtain list of confirmed attributes

getSelectedAttributes(final.boruta,withTentative = F)

## Checking the results in data frame

boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)

