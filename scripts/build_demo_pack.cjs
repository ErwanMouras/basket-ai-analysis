// Compose an editorial presentation around captured, unchanged viewer outputs.
const fs = require('node:fs/promises');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');
const {createHash} = require('node:crypto');
const {chromium} = require('../viewer/node_modules/playwright');
const root = path.resolve(__dirname,'..');
const args = process.argv.slice(2);
const output = path.resolve(args.includes('--output') ? args[args.indexOf('--output')+1] : path.join(root,'release/demo-2026-10-27'));
const esc = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ffmpeg = args => execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y',...args],{stdio:['ignore','inherit','inherit']});
function body(slide, index, landscape = false) {
  return `<section class="slide ${landscape?'landscape':''}" id="slide-${index+1}">
    <header><b>VYNEROS</b><span>COMPUTER VISION / BASKETBALL</span></header>
    <div class="layout"><div class="intro"><p class="eyebrow">${slide.tag}</p><h1>${slide.title}</h1><p class="subtitle">${slide.subtitle}</p></div>
    <div class="visual">${slide.visual}</div></div>
    <footer><span>${slide.footer}</span><b>${String(index+1).padStart(2,'0')} / 06</b></footer>
  </section>`;
}
function html(slides, landscape = false) {
  const font = pathToFileURL(path.join(root,'viewer/BarlowCondensed-Bold.ttf')).href;
  return `<!doctype html><html lang="fr"><meta charset="utf-8"><title>Vyneros — du match aux données</title>
  <style>
  @font-face{font-family:Barlow;src:url('${font}');font-weight:700}*{box-sizing:border-box}
  @page{size:1080px 1350px;margin:0}body{margin:0;background:#101113;color:#f5f2ed;font-family:Arial,sans-serif}
  .slide{width:1080px;height:1350px;padding:56px 60px 42px;position:relative;overflow:hidden;background:radial-gradient(ellipse at 100% 0,#f28b500d,transparent 50%),#111214;break-after:page}
  .slide:last-child{break-after:auto}header,footer{display:flex;justify-content:space-between;align-items:center}
  header{height:36px;border-bottom:1px solid #3c3835;padding-bottom:20px}header b{font:700 30px Barlow;letter-spacing:3px;color:#f28b50}header span{font-size:13px;letter-spacing:2px;color:#aaa6a1}
  .eyebrow{color:#f28b50;font-size:17px;letter-spacing:3px;margin:40px 0 15px;text-transform:uppercase}
  h1{font:700 106px/.96 Barlow;margin:0 0 24px;letter-spacing:-1px}h1 em{color:#f28b50;font-style:normal}
  .subtitle{color:#c5c0b9;font-size:28px;line-height:1.4;margin:0 0 30px;max-width:930px}
  .media{width:960px;height:540px;object-fit:contain;background:#171a1d;border:1px solid #393b3e;border-radius:10px;display:block}
  .ui{width:960px;max-height:520px;object-fit:contain;border:1px solid #393b3e;border-radius:10px;display:block;background:#171a1d}
  .row{width:100%;display:block;border-radius:6px;margin:20px 0}
  .pills{display:flex;gap:12px;margin-top:28px}.pills span{padding:16px 22px;border:1px solid #694c37;border-radius:7px;color:#ffc299;font-size:22px;flex:1;text-align:center;background:#30251c}
  .note{font-size:23px;line-height:1.45;color:#b8b1a9;margin:22px 0 0}.note strong{color:#f5f2ed;font-weight:400}
  .maps{display:grid;grid-template-columns:1fr 1fr;gap:20px}.map{padding:18px;border:1px solid #393b3e;border-radius:10px;background:#1a1b1e}
  .map h2{font-size:26px;font-weight:400;margin:0 0 16px}.map img{width:420px;height:430px;object-fit:contain;display:block}.map p{font-size:19px;line-height:1.4;color:#b8b1a9}
  .numbers{display:flex;gap:16px;margin:24px 0}.numbers div{flex:1;padding:20px;border:1px solid #3d3732;border-radius:8px;background:#1a1b1e}.numbers b{font:700 53px Barlow;color:#f5f2ed;display:block}.numbers span{font-size:19px;line-height:1.4;color:#b8b1a9}
  .pipeline{display:grid;gap:12px}.step{border:1px solid #574534;border-radius:8px;padding:22px 26px;background:#211d19;display:flex;gap:24px;align-items:center}.step b{font:700 36px Barlow;color:#f28b50}.step strong{font-size:27px;font-weight:400}.step small{display:block;color:#bab2a8;font-size:21px;margin-top:8px}
  .statuses{display:grid;grid-template-columns:1fr 1fr;gap:20px}.status{padding:18px;border:1px solid #3d3732;border-radius:8px;background:#1a1b1e}.status h2{font-size:25px;font-weight:400;margin:0 0 12px;color:#d5b9f5}.status img{width:100%;height:90px;object-fit:contain}.status p{font-size:20px;color:#b8b1a9;margin-bottom:0}
  .choices{display:grid;gap:18px;margin:28px 0}.choice{font-size:34px;padding:24px 28px;border:1px solid #694c37;border-radius:8px;background:#211d19}.choice span{color:#f28b50;margin-right:20px}
  .repo{font-size:27px;line-height:1.5;padding:26px;border-left:3px solid #f28b50;margin-top:32px;overflow-wrap:anywhere}.repo small{display:block;color:#b8b1a9;font-size:22px}
  footer{position:absolute;left:60px;right:60px;bottom:36px;border-top:1px solid #3c3835;padding-top:18px;font-size:16px;letter-spacing:.3px;color:#aaa6a1}footer b{font-weight:400;color:#f28b50}
  .landscape{width:1920px;height:1080px;padding:50px 60px}.landscape .layout{display:grid;grid-template-columns:620px 1fr;gap:80px;align-items:center;height:850px}.landscape h1{font-size:104px}.landscape .subtitle{font-size:28px}.landscape .eyebrow{margin-top:0}
  .landscape .media,.landscape .ui{width:1100px;height:619px;max-height:619px}.landscape .maps{gap:22px}.landscape .map img{width:490px;height:520px}.landscape .row{margin:16px 0}.landscape .numbers b{font-size:50px}.landscape footer{bottom:35px}.landscape .note{font-size:23px}
  .landscape .ui{height:500px;max-height:500px}
  </style>${slides.map((s,i)=>body(s,i,landscape)).join('')}</html>`;
}
function captions() {
  const lines=[
    'D’une vidéo de basket à des actions que l’on peut explorer.',
    'Un clic revient avant le départ estimé du tir, en pause.',
    'Présence et origines des tirs : deux lectures du même extrait.',
    'Derrière le viewer : perception, suivi, géométrie et artefacts.',
    'Les propositions incertaines restent à vérifier.',
    'Sur un nouveau match, quel serait votre premier test de robustesse ?',
  ];
  return lines.map((line,i)=>`${i+1}\n00:00:${String(i*6).padStart(2,'0')},000 --> 00:00:${String((i+1)*6).padStart(2,'0')},000\n${line}\n`).join('\n');
}
(async()=>{
  const manifest=JSON.parse(await fs.readFile(path.join(output,'capture-manifest.json'),'utf8'));
  if(!manifest.real||!manifest.public)throw Error('Build requires an all-mode capture manifest');
  const lal=manifest.real.team_maps.find(m=>m.team_id==='los_angeles_lakers');
  const frames=manifest.real.frames.toLocaleString('fr-FR');
  const seconds=manifest.real.timings.processing_s.toLocaleString('fr-FR',{maximumFractionDigits:2,minimumFractionDigits:2});
  const slides=[
    {tag:'Une action. Plusieurs lectures.',title:'Du match<br>aux <em>données.</em>',subtitle:'Une vidéo de basket devient un point de départ pour explorer le jeu.',
      visual:'<img class="media" src="real/annotated.png" alt="Image réelle du match avec les calques du viewer"><div class="pills"><span>Suivre</span><span>Projeter</span><span>Revoir</span></div><p class="note">Images de développement · <strong>analyse calculée avant la lecture</strong></p>',footer:'Vyneros · pipeline de vision + viewer'},
    {tag:'01 / Relecture',title:'Un tir devient<br>un <em>point d’entrée.</em>',subtitle:'« Voir le tir » ramène au contexte avant le relâchement estimé.',
      visual:'<img class="ui" src="real/review-start.png" alt="Viewer en pause avant le départ du tir"><img class="row" src="real/probable-row.png" alt="Ligne du tir probable de Luke Kennard"><div class="numbers"><div><b>16,80 s</b><span>départ estimé</span></div><div><b>504</b><span>frame source</span></div><div><b>16,20 s</b><span>relecture en pause</span></div></div>',footer:'Proposition probable · aucun résultat marqué / raté attribué'},
    {tag:'02 / Lecture collective',title:'Lire <em>l’espace.</em><br>Retrouver les tirs.',subtitle:'Présence filtrée par situation. Carte des tirs sur tout l’extrait.',
      visual:`<div class="maps"><div class="map"><h2>Présence · LAL en attaque</h2><img src="real/presence.png" alt="Heatmap de présence LAL"><p>Intervalles observés en jeu installé.</p></div><div class="map"><h2>Origines des tirs · LAL</h2><img src="real/shots.png" alt="Carte des origines des tirs LAL"><p>● Probable &nbsp; ○ À vérifier</p></div></div><div class="numbers"><div><b>${lal.probable_count}</b><span>probables attribués</span></div><div><b>${lal.candidate_count}</b><span>candidat attribué</span></div><div><b>${lal.located_count}/${lal.detected_count}</b><span>origines localisées</span></div></div><p class="note">Des propositions sur cet extrait, pas un total officiel du match.</p>`,footer:'Positions au sol des tireurs · ballon aérien non projeté au sol'},
    {tag:'03 / Ingénierie',title:'Un pipeline complet.<br>Des sorties <em>inspectables.</em>',subtitle:'Relier les modèles à des données que l’on peut réellement explorer.',
      visual:`<div class="pipeline"><div class="step"><b>01</b><div><strong>Perception</strong><small>Personnes · ballon · pose · maillots</small></div></div><div class="step"><b>02</b><div><strong>Suivi et identité</strong><small>Tracks · effectif · conflits d’identité</small></div></div><div class="step"><b>03</b><div><strong>Géométrie et mesures</strong><small>Terrain · intervalles valides · couverture</small></div></div><div class="step"><b>04</b><div><strong>Artefacts et viewer</strong><small>Résultats enregistrés · relecture · tests</small></div></div></div><div class="numbers"><div><b>${frames}</b><span>frames analysées</span></div><div><b>${seconds} s</b><span>traitement batch du run</span></div></div><p class="note">Profil E-BARD + SAM / YOLO26L ballon · RTX 5080.<br>La vitesse de lecture ne mesure pas le débit d’inférence.</p>`,footer:'DVC · MLflow · configurations figées · Docker · CI'},
    {tag:'04 / Revue des résultats',title:'L’incertitude<br>reste <em>visible.</em>',subtitle:'Un probable et un candidat conservent leurs statuts dans le viewer.',
      visual:'<div class="statuses"><div class="status"><h2>Probable</h2><img src="real/probable-row.png" alt="Tir probable à 16,80 s"><p>16,80 s · frame 504</p></div><div class="status"><h2>À vérifier</h2><img src="real/candidate-row.png" alt="Candidat à vérifier à 47,57 s"><p>47,57 s · frame 1 427</p></div></div><img class="media" style="margin-top:20px;height:400px" src="real/candidate.png" alt="Image du candidat dans la vidéo"><p class="note">Des niveaux de support heuristiques.<br><strong>Pas des probabilités calibrées.</strong></p>',footer:'Exemples de développement · généralisation indépendante à évaluer'},
    {tag:'Et sur un nouveau match ?',title:'Quel serait votre<br>premier test<br>de <em>robustesse ?</em>',subtitle:'Trois angles pour challenger la suite du projet.',
      visual:'<div class="choices"><div class="choice"><span>01</span> L’identité des joueurs</div><div class="choice"><span>02</span> La géométrie du terrain</div><div class="choice"><span>03</span> Les changements de caméra</div></div><div class="repo">github.com/ErwanMouras/<br>basket-ai-analysis<small>Code, protocoles, résultats et limites.</small></div><p class="note">Le sample public permet d’essayer l’interface sans GPU.<br>Il est synthétique et scripté.</p>',footer:'Vyneros · retours techniques bienvenus'},
  ];
  await fs.writeFile(path.join(output,'carousel.html'),html(slides));
  await fs.writeFile(path.join(output,'readme-master.html'),html(slides,true));
  const srt=captions();
  await fs.writeFile(path.join(output,'linkedin-36s.fr.srt'),srt);
  await fs.writeFile(path.join(output,'linkedin-36s.fr.vtt'),'WEBVTT\n\n'+srt.replace(/(\d{2}:\d{2}:\d{2}),(\d{3})/g,'$1.$2'));
  const browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  const social=[],wide=[],mediaPositions={};
  try {
    for(const landscape of [false,true]) {
      const page=await browser.newPage({viewport:landscape?{width:1920,height:1080}:{width:1080,height:1350},deviceScaleFactor:1});
      await page.goto(pathToFileURL(path.join(output,landscape?'readme-master.html':'carousel.html')).href);
      await page.evaluate(()=>document.fonts.ready);
      const overflow=await page.locator('.slide').evaluateAll(slides=>slides.map((slide,index)=>({page:index+1,
        visualBottom:slide.querySelector('.visual').getBoundingClientRect().bottom,
        footerTop:slide.querySelector('footer').getBoundingClientRect().top,
        scrollWidth:slide.scrollWidth,clientWidth:slide.clientWidth,
      })).filter(item=>item.visualBottom>item.footerTop-8||item.scrollWidth>item.clientWidth));
      if(overflow.length)throw Error(`Presentation layout overflow (${landscape?'16:9':'4:5'}): ${JSON.stringify(overflow)}`);
      for(let index=0;index<slides.length;index++) {
        const file=path.join(output,`${landscape?'readme':'carousel'}-${String(index+1).padStart(2,'0')}.png`);
        await page.locator(`#slide-${index+1}`).screenshot({path:file});
        (landscape?wide:social).push(file);
      }
      const media=await page.locator('#slide-1 .media').boundingBox();
      const first=await page.locator('#slide-1').boundingBox();
      mediaPositions[landscape?'readme':'social']={x:Math.round(media.x-first.x),y:Math.round(media.y-first.y),width:Math.round(media.width),height:Math.round(media.height)};
      if(!landscape) await page.pdf({path:path.join(output,'carousel.pdf'),preferCSSPageSize:true,printBackground:true});
      await page.close();
    }
  } finally {await browser.close();}
  await fs.copyFile(social[0],path.join(output,'linkedin-cover.png'));
  for(const [kind,files] of [['social',social],['readme',wide]]) {
    const position=mediaPositions[kind];
    const segments=[];
    for(let index=0;index<files.length;index++) {
      const file=path.join(output,`${kind}-segment-${index+1}.mp4`);
      const inputs=['-loop','1','-framerate','30','-i',files[index]];
      const filters=[];
      if(index===0) {
        inputs.push('-i',path.join(output,'real/raw.mp4'),'-i',path.join(output,'real/annotated.mp4'));
        filters.push('-filter_complex',`[1:v]scale=${position.width}:${position.height},setsar=1,tpad=stop_mode=clone:stop_duration=2.4[raw];[2:v]scale=${position.width}:${position.height},setsar=1,tpad=stop_mode=clone:stop_duration=2.4,format=rgba,fade=t=in:st=1:d=0.3:alpha=1[annotated];[raw][annotated]overlay=0:0[comparison];[0:v][comparison]overlay=${position.x}:${position.y}:shortest=1[out]`,'-map','[out]');
      }
      ffmpeg([...inputs,...filters,'-t','6','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',file]);
      segments.push(file);
    }
    const list=path.join(output,`${kind}-concat.txt`);
    await fs.writeFile(list,segments.map(file=>`file '${file.replace(/'/g,"'\\''")}'`).join('\n')+'\n');
    ffmpeg(['-f','concat','-safe','0','-i',list,'-c','copy','-movflags','+faststart',path.join(output,kind==='social'?'linkedin-36s.mp4':'readme-demo-36s.mp4')]);
  }
  // Only synthetic screenshots are copied into the public repository.
  const publicDir=path.join(root,'imagesReadMe');
  for(const name of ['playback','player','team'])await fs.copyFile(path.join(output,'public',`${name}.png`),path.join(publicDir,`viewer-2026-10-27-${name}.png`));
  const tour=path.join(output,'public-tour.txt');
  const tourFiles=['playback','player','team','layers'];
  await fs.writeFile(tour,tourFiles.map(name=>`file '${path.join(output,'public',`${name}.png`)}'\nduration 2`).join('\n')+`\nfile '${path.join(output,'public/layers.png')}'\n`);
  ffmpeg(['-f','concat','-safe','0','-i',tour,'-filter_complex','fps=8,scale=960:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3','-t','8','-loop','0',path.join(publicDir,'viewer-2026-10-27-tour.gif')]);
  const gallery=`<!doctype html><html lang="fr"><meta charset="utf-8"><title>Vyneros — kit de démonstration</title><style>body{background:#111214;color:#f3f0ea;font:18px/1.6 Arial;max-width:1180px;margin:40px auto;padding:24px}h1{color:#f28b50}a{color:#ffc299}video{max-width:100%;max-height:740px;background:#171a1d}img{width:calc(33% - 16px);margin:8px;vertical-align:top}nav{display:flex;gap:24px;flex-wrap:wrap}</style><h1>Vyneros — du match aux données</h1><p>Proposition de présentation · images de développement · traitement batch. Aucun média n’a été publié.</p><nav><a href="carousel.pdf">Carrousel PDF</a><a href="linkedin-cover.png">Miniature</a><a href="linkedin-36s.fr.srt">Sous-titres</a><a href="capture-manifest.json">Provenance des captures</a><a href="readme-demo-36s.mp4">Master 16:9</a></nav><h2>LinkedIn · 36 secondes</h2><video controls poster="linkedin-cover.png" src="linkedin-36s.mp4"><track kind="subtitles" src="linkedin-36s.fr.srt" srclang="fr" label="Français"></video><h2>Pages du carrousel</h2>${social.map(file=>`<a href="${esc(path.basename(file))}"><img src="${esc(path.basename(file))}" alt="Page du carrousel"></a>`).join('')}<h2>Extrait réel annoté · 3,6 secondes</h2><video controls src="real/annotated.mp4"></video><h2>README public · sample synthétique scripté</h2><p>Ces aperçus montrent le viewer. Ils ne sont pas des prédictions de modèles.</p>${['playback','player','team'].map(name=>`<a href="public/${name}.png"><img src="public/${name}.png" alt="Aperçu synthétique ${name}"></a>`).join('')}</html>`;
  const finalGallery=gallery.replace('src="linkedin-36s.fr.srt"','src="linkedin-36s.fr.vtt"')
    .replace('<h2>Pages du carrousel</h2>','<h2>Capture continue du viewer</h2><p>Interactions réelles, résultats calculés en amont.</p><video controls src="viewer-walkthrough.mp4"></video><h2>Pages du carrousel</h2>');
  await fs.writeFile(path.join(output,'index.html'),finalGallery);
  const videos={};
  for(const name of ['linkedin-36s.mp4','readme-demo-36s.mp4','real/annotated.mp4',...(manifest.walkthrough?['viewer-walkthrough.mp4']:[])]) {
    videos[name]=JSON.parse(execFileSync('ffprobe',['-v','error','-show_entries','format=duration,size:stream=width,height,codec_name,r_frame_rate','-of','json',path.join(output,name)],{encoding:'utf8'}));
  }
  const hashes={};
  for(const name of [...Object.keys(videos),'carousel.pdf','linkedin-cover.png','linkedin-36s.fr.srt'])
    hashes[name]=createHash('sha256').update(await fs.readFile(path.join(output,name))).digest('hex');
  await fs.writeFile(path.join(output,'media-validation.json'),JSON.stringify({videos,sha256:hashes,social_pages:social.length,readme_pages:wide.length,public_media:'Synthetic scripted sample only',capture_manifest:'capture-manifest.json'},null,2)+'\n');
  console.log(`Demo pack: ${path.join(output,'index.html')}`);
})().catch(error=>{console.error(error);process.exitCode=1;});
