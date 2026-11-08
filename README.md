# Etkinlik arşivi

Etkinlik fotoğrafları ve videoları bu bilgisayarda duruyor. Araç, sizin vereceğiniz tanıtım fotoğraflarındaki yüzleri arar ve eşleşen dosyaları kişi klasörlerine **kopyalar**. Kaynak silinmez. Fotoğraflar hiçbir yere yüklenmez.

Yüz verisi kişisel veridir. Yalnızca buna izin veren kişileri tanıtın. Kayıtlı kişiler `veri/kisiler/` altındadır ve depoya girmez. Fotoğraflar, videolar, modeller ve çıktılar da depoya girmez.

Araç, Kadir Has Üniversitesi Silivri Teknopark'taki bir hackathon'un fotoğrafları için yazıldı. O kaynak klasörde 628 `.jpg` ve 188 `.mp4` vardı. Bu `.mp4`'lerin 132'si 0 bayttı; kopyalama yarım kalmıştı. Bazı klasörler aynı kareleri tekrar içeriyordu: dolu 684 dosyanın 323'ü benzersizdi. Aşağıdaki sayılar o etkinlikten.

Etkinlik klasörünü proje köküne koyun; panel en çok fotoğraf ve video içeren klasörü kaynak olarak önerir. Başka bir yerdeyse panelden seçin. Çıktı varsayılan olarak proje kökündeki `cikti/` klasörüdür ve kaynak klasörün içinde olamaz.

## Tekil havuz

```bash
uv run recognize havuz
```

Etkinlik klasöründeki her farklı fotoğraf ve videodan birer tane `tekil-havuz/foto` ve `tekil-havuz/video` içine yazılır: 295 fotoğraf, 26 video. Birebir kopyalar, 0 baytlık dosyalar ve açılmayan iki `.mp4` alınmaz; hepsi `tekil-havuz/havuz.json` içinde listelenir. Dosyalar APFS klonudur, diskte ikinci kez yer kaplamaz. Asıl klasöre dokunulmaz. Havuz varsa panel ve tarama onu kaynak olarak kullanır. Panelde aynı iş «Tekil havuz oluştur» düğmesiyle yapılır.

## Kurulum

```bash
uv sync
uv run recognize panel
```

Panel `http://127.0.0.1:8765` adresinde açılır. Yüz bulma için YuNet (MIT), yüz tanıma için SFace (Apache 2.0) kullanılır; ikisi de OpenCV Zoo'dan gelir. İki model ilk kullanımda `modeller/` klasörüne indirilir.

## Panel

Panel üç bölümden oluşur: **Sonuçlar**, **Kişiler**, **Ayırma**. Açılışta son ayırmanın sonuçları gelir (`veri/son-taramalar.json` en son çıktı klasörünü hatırlar; komut satırından yapılan taramalar da listeye girer).

- **Sonuçlar:** kişi başına küçük resimler, «Tümünü gör» ile tam liste, «Finder'da göster» (Windows'ta Gezgin). Küçük yüz, ikinci bakış, kıyafet desteği ve eşiğe yakın eşleşmeler «kontrol edilmeli» diye işaretlenir; «Kontrol et» süzgeci yalnızca bunları gösterir. Büyük önizlemede eşleşen yüz çerçeve içindedir.
- **Bu kişi değil:** yanlış eşleşme kişinin klasöründen `_Yanlış eşleşmeler/<Kişi>/` altına taşınır; fotoğraf silinmez, kaynak klasöre dokunulmaz, `rapor.json` ve sayımlar güncellenir. «Geri al» dosyayı yerine koyar; yeniden yükleme ya da panel yeniden başlasa da «Ayıklananlar» listesinden geri alınabilir.
- **Önceki taramalar:** Aynı çıktıya yeniden taranınca kişi klasöründe eski taramadan kalan dosyalar gösterilir. «Önceki taramaya taşı» bunları `_Önceki taramalar/<Kişi>/` altına alır; hiçbiri silinmez. Tarama bitmeden bu dosyalar listelenmez.
- **Durdur:** süren ayırma dosya sınırında durur; o ana kadar işlenen dosyalar kişi klasörlerinde kalır ve rapor `complete: false` ile yazılır. Rapor zaten ilk kopyadan sonra ve her 25 dosyada bir ara kayıt olarak yazılır; yeni bir tarama eski `rapor.json`'u `veri/rapor-arsivi/` altına kopyalar (son 10 tutulur).
- **İlerleme:** aşama, yüzde, tahmini süre (videolar örneklenen kare sayısıyla ağırlıklı), kişi başına canlı sayım ve son bulunan kare. Sekme başlığı yüzdeyi gösterir; sayfa yenilense de süren iş kaldığı yerden izlenir.
- **Kişiler:** silmeden önce onay sorulur. Silme hemen uygulanır ve hemen çıkan «Geri al» kişiyi geri getirir (`veri/silinenler/`, 7 gün sonra temizlenir). Silinen kişinin fotoğrafları etkinlik arşivinde görünmez; dosyalar çıktı klasöründe kalır. «Kişi ekle» kamerayı boş adla açar; kayıtlı bir ad yazılırsa yeni açıların o kişiye ekleneceği söylenir. Bozuk bir kayıt diğerlerini gizlemez; panelde uyarı olarak görünür ve silinebilir.
- **Tema:** sağ üstteki ay ya da güneş düğmesi açık ve koyu tema arasında geçer. Seçim bu tarayıcıda saklanır; hiç seçilmediyse sistemin ayarı izlenir.
- **Güvenlik:** panel yalnızca `127.0.0.1`, `localhost` ve `::1` adreslerinden yanıt verir; başka sitelerden gelen istekler reddedilir; sayfa çerçeve içine alınamaz. Küçük resimler yalnızca raporun listelediği dosyalardan üretilir (`veri/onizleme/` önbelleği, en çok 512 MB).

## Kişi tanıtma

Aynı kişi her karede karşıdan bakmaz. Önden, soldan, sağdan ve gözleri kapalı birer fotoğraf ekleyin. Her yüz ayrı saklanır; eşleşme bu karelerin **en yakın olanına** bakar, ortalamasına değil. Tek önden fotoğrafla da kaydedilir. Yan açılar o zaman daha sık kaçar.

Bu etkinlikte bir kişi baştan sona sarı giymişti. Diğerleri tek tip giymedi; onlara kıyafet rengi vermeyin. Sarı, kimliği tek başına seçmez. Yüz skoru eşiğin hemen altındaysa ve çenenin altındaki bölge o renge uyuyorsa desteğe girer.

```bash
uv run recognize tanit "Ad Soyad" on.jpg sol.jpg sag.jpg goz.jpg --kiyafet sari
uv run recognize tanit "Başka Kişi" on.jpg sol.jpg sag.jpg
uv run recognize liste
```

Grup fotoğrafında doğru yüze tıklamak için paneli kullanın. «Bu etkinlikte belirgin kıyafet» listesini yalnızca gerçekten tek renk giyen kişi için seçin.

Aynı adla yeniden `tanit` derseniz yeni kareler eklenir, eskiler durur.

### Kamerayla tanıtma

Panelde «Kamerayla tanıt» (veya `uv run recognize kamera "Ad Soyad"`). Yüzünüz çemberin içinde kendiliğinden ortalanır. Önce düz bakarak yüzün dinlenme açısı öğrenilir, sonra başınızı yavaşça çevirirsiniz. Çemberin çevresindeki çizgiler başınızın döndüğü yöne uzar; o yönden yeterli açı alınınca o kesim yeşile döner. Sol 4, sağ 4, yukarı 3, aşağı 3 ve düz 2 açı, toplam 16 bakış alınınca kayıt kendiliğinden yazılır. Işık, bulanıklık, uzaklık veya kadrajdaki ikinci yüz gibi bir sorun, ekranda yarım saniyeden uzun sürerse uyarı olarak gösterilir; tek bir bozuk kare yazıyı titretmez.

Bir kişiye en fazla 96 referans yazılır. Yer kalmadıysa kamera açılmadan söylenir; kişiyi silip yeniden tanıtın.

## Ayırma

Kişiler kaydedildikten sonra. Varsayılan eşik 0,50. Kayıtlı yan açılar bu çizginin üstünde kalır; 0,36 kalabalıkta başka insanları da alır. Videoda 0,5 saniyede bir kareye bakılır. Kayıtlı herkes bir videoda doğrulanınca o video erken biter.

```bash
uv run recognize tara "etkinlik-klasoru" --cikti cikti
```

Desteklenen dosyalar: jpg, png, webp ve heic fotoğraf; mp4, mov, m4v, avi ve mkv video. Klasör düzeni önemli değil: bütün alt klasörler gezilir, fotoğraf ve video karışık olabilir. iPhone'un HEVC kodlu .mov videoları da okunur.

Ayırmadan önce «Kimler için ayrılsın?» kısmından bir, iki ya da istediğiniz kayıtlı kişileri seçin (komut satırında `--kisi "Ad Soyad"`, birden çok kez verilebilir). Her yüz yine bütün kayıtlı kişilerle karşılaştırılır; yalnızca seçilenlerin klasörü yazılır. Böylece seçilmeyen birinin yüzü seçilen kişiye verilmez.

Panelde kaynak ve çıktı klasörü yazılmaz, seçilir: karta tıklayınca macOS'ta Finder, Windows'ta Gezgin klasör penceresi açılır. Pencere açılamazsa (ör. uzak bağlantı) yol yazma kutusu çıkar. Panel, başka bir siteden gelen tarama ya da klasör isteğini reddeder.

Dosya kopyalanır. Aynı karede iki kişi varsa dosya iki klasöre de yazılır. Kaynağı silmek için `--tasi`. Raporda `clothingAssisted: true`, zayıf yüzün kıyafet rengiyle desteklendiği anlamına gelir.

Birebir aynı içerikteki dosyalar bir kez incelenir ve kişi klasörüne bir kez yazılır; kopyalar raporda `duplicateOf` ile ilk dosyayı gösterir. `--tasi` seçiliyse taşınan dosyanın kopyaları da kaynaktan silinir. 0 baytlık dosyalar hata sayılmaz, `skipped` ile atlanır.

İkinci bakış: tarama bitince, kişinin etkinlikte çok net görüldüğü yüzler (skor en az 0,60, yüz en az 48 piksel) o günün ışığını, saçını ve kamerasını taşıdığı için geçici referans olur ve fotoğraflara bir kez daha bakılır. Bu yüz kendi fotoğrafına kefil olmaz, 32 pikselden küçük yüzlere uygulanmaz ve kayıtlı kişiye yazılmaz. Bu etkinlikte 16 kamera açısıyla 53 olan sonuç 63'e çıktı; eklenen 10 fotoğrafın hepsi elle kontrol edildi. Raporda bu eşleşmeler `learned: true` ile işaretlidir.

Her eşleşmede `facePx` yüzün kısa kenarıdır. 32 pikselden küçük yüzlerde `smallFace: true` yazılır ve panel bunları ayrıca sayar: bu boyutta benzer yüzlü biri de 0,50'yi geçebilir. Bu etkinlikte yanlış giden iki eşleşmenin ikisi de 24–25 piksellik yüzlerdi.

## Sınırlar

Çok küçük, arkası dönük veya karanlık yüzler kaçabilir. Fotoğraflar 1920 piksele küçültülüp taranır. 3200 piksel denendi: 20 fotoğraf daha buldu ama bunların çoğu, arka sıradaki 26–43 piksellik yüzlerde başka bir kişiydi. Bu yüzden 1920'de kalındı. Kapalı göz ve yan profil özel olarak elenmez. Kıyafet rengi yalnızca o kişiye kayıtlıysa devreye girer. Karede sarı bir şey olması, o kişinin yüzü yoksa dosyayı klasöre koymaz.

## Lisans

MIT, bkz. [LICENSE](LICENSE). Modeller bu depoda yoktur, kendi lisanslarıyla indirilir: YuNet MIT, SFace Apache 2.0.
