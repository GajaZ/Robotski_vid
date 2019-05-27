# Robotski_vid
Projekt pri predmetu Robotski vid

Algoritem uporablja Meanshift algoritem za iskanje lokalnih optimumov v funkcijah gostote verjetnosti z fiksnim oknom. Gostote verjetnosti določimo na podlagi frekvenčne analize pojavnosti značilnic, tipično so to barve ali teksture objektov zanimanja. Pri-lagoditev Meanshift algoritma za namen sledenja objektov v videu je v adaptivnivelikosti okna.

[referenca](https://docs.opencv.org/3.4.1/db/df8/tutorial_py_meanshift.html)


Komentar:
CAMShifta mi ni uspelo do konca implementirati. Zaenkrat od tracker-jev, ki so v kodi na voljo, deluje MeanShiftHSV (zgledno) in MeanShiftRGB (zelo počasi).
Na začetku je najprej potrebno izbrati vrsto algoritma (izbran je MeanShiftHSV) in video ter pripadajoči track_window, ki bi ga želeli testirati.
