@echo off
rem BeamNG.Earth with Sofia Municipality's own 1 m terrain model (server/sofiadtm.py).
rem Close the BeamNG.Earth app and BeamNG.drive first, double-click this file, then start BeamNG.drive.
rem The file is Sofiaplan's dtm_so_1m_cog_2024.tif (https://sofiaplan.bg/api/, dataset 656), unzipped.
set "BNE_SOFIA_DTM=E:\beamEarth\sofia\dtm_so_1m_cog_2024.tif"
if not exist "%BNE_SOFIA_DTM%" (
  echo The terrain file is missing: %BNE_SOFIA_DTM%
  echo Sofia would load with the usual 30 m heights. Fix the path in run_sofia.bat.
  pause
  exit /b 1
)
rem Sofia's streets as surveyed (server/sofiastreets.py): GIS-Sofia's "Streets and sidewalks" (streets_curbs.geojson,
rem https://urbandata.sofia.bg/dataset/streets_curbs-zip, unzipped). Without the file: the roads' guessed widths.
set "BNE_SOFIA_STREETS=E:\beamEarth\sofia\streets\streets_curbs.geojson"
if not exist "%BNE_SOFIA_STREETS%" (
  echo The streets file is missing: %BNE_SOFIA_STREETS%
  echo Sofia's roads keep their guessed widths.
  set "BNE_SOFIA_STREETS="
)
rem GIS-Sofia's 2024 aerial photo as the ground's texture in Sofia (server/imagery_regional.py SOFIA_ORTHO): fetched
rem from the municipality's server while you drive, kept in the cache. Put "rem " before the next line to turn it off
rem (the satellite photo as before), for example if that server is slow or down.
set "BNE_SOFIA_ORTHO=on"
call "%~dp0run_server.bat" %*
