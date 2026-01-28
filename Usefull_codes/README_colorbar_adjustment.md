# Guide d'ajustement des colorbars DoLP

## Problème corrigé

Pour certaines étoiles (R_Hya, AK_Hya, BK_Vir), les colorbars DoLP affichaient des couleurs (notamment le jaune) qui n'apparaissaient pas dans l'image visible. 

**Cause**: La colorbar utilisait les valeurs min/max de toute l'image `sub_v_dolp_display` avant tout rognage ou masquage, incluant potentiellement des pixels aberrants ou hors de la zone d'intérêt finale.

## Solution implémentée

Un nouveau dictionnaire `custom_dolp_vrange` a été ajouté pour personnaliser les limites vmin/vmax de la colorbar par étoile.

### Localisation dans le code

Ligne ~425 dans `polarized_maps_paper&_thesis.py`:

```python
# Dictionnaire pour personnaliser les limites vmin/vmax de la colorbar DoLP
# Utile quand des pixels hors zone d'intérêt faussent l'échelle
custom_dolp_vrange = {
    'R_Hya': {'vmin': None, 'vmax': 0.04},     # Limite le max à 0.04
    'AK_Hya': {'vmin': None, 'vmax': 0.025},   # Limite le max à 0.025
    'BK_Vir': {'vmin': None, 'vmax': 0.025},   # Limite le max à 0.025
}
```

### Comment ajuster pour d'autres étoiles

1. **Identifier le problème**: La colorbar montre des couleurs absentes de l'image

2. **Déterminer les bonnes limites**:
   - Examiner les niveaux de contours personnalisés dans `custom_dolp_contours`
   - Le vmax devrait généralement être proche du niveau de contour le plus élevé
   - Exemple: si les contours vont de 0.016 à 0.03, mettre vmax=0.04 ou 0.035

3. **Ajouter l'étoile au dictionnaire**:
   ```python
   custom_dolp_vrange = {
       'Nom_Etoile': {'vmin': None, 'vmax': 0.XX},  # vmin=None utilise le min automatique
       # ou
       'Autre_Etoile': {'vmin': 0.01, 'vmax': 0.05},  # Limites min et max personnalisées
   }
   ```

4. **Relancer le script**: Les nouvelles limites seront automatiquement appliquées

## Exemples de valeurs typiques

- **Étoiles avec forte polarisation**: vmax ~ 0.1 - 0.3
- **Étoiles avec polarisation modérée**: vmax ~ 0.03 - 0.06
- **Étoiles avec faible polarisation**: vmax ~ 0.02 - 0.04

## Notes techniques

- `vmin=None` utilise le minimum automatique de l'image (généralement ~0)
- `vmax=None` utilise le maximum automatique de l'image
- Les deux limites sont utilisées dans 2 endroits:
  1. Panneau combiné DoLP+PI (ligne ~825)
  2. Figure DoLP séparée (ligne ~1211)- **Important**: Les contours utilisent toujours les vraies valeurs min/max de l'image, seule la colorbar est limitée

## Détails d'implémentation

Le code gère automatiquement les valeurs `None`:
```python
vmin_dolp = vrange.get('vmin') if vrange.get('vmin') is not None else np.nanmin(sub_v_dolp_display)
vmax_dolp = vrange.get('vmax') if vrange.get('vmax') is not None else np.nanmax(sub_v_dolp_display)
```

Pour les contours (qui doivent couvrir toute la gamme de l'image):
```python
dolp_min_real = np.nanmin(sub_v_dolp_display)  # Toujours les vraies valeurs
dolp_max_real = np.nanmax(sub_v_dolp_display)
contour_levels = custom_dolp_contours.get(star_name, np.linspace(dolp_min_real, dolp_max_real, 5))
```
## Vérification

Après modification, vérifier que:
- ✅ Toutes les couleurs de la colorbar apparaissent dans l'image
- ✅ Les contours blancs correspondent bien aux valeurs de la colorbar
- ✅ L'image n'est ni trop saturée ni trop sombre
