
async function loadArrayFromURL(fileName, opts){
    var buffer = new Float32Array(await loadBuffer(fileName, opts))
    var shape = fileName.match(/\d+(x\d+)*$/)[0].split('x').map(k => +k)
    return ndarray(buffer, shape)
}

async function loadBuffer(fileName, {progressContainer}={}){
    var xhr = new XMLHttpRequest()
    xhr.open('GET', fileName, true)
    xhr.responseType = 'arraybuffer'
    xhr.send(null)

    var prog = createProgress('downloading model', progressContainer)
    xhr.onprogress = function(progress){
        prog.value = progress.loaded / progress.total
    }

    await new Promise(resolve => xhr.onload = resolve)
    prog.destroy()
    
    return xhr.response;
}

function createProgress(label, container){
    // document.querySelector('.pie-wrapper').style.display = ''

    // document.querySelector('.pie-wrapper .label').innerText = label;
    return {
        set value(n){
            // document.querySelector('.pie .left-side').style.transform = 'rotate(' + (360 * n) + 'deg)'
            // document.querySelector('.pie').classList.toggle('over-fifty', n > 0.5)
            var p = Math.round(n * 100);
            document.getElementById('progress').style.background = '-webkit-linear-gradient(left, rgb(204, 208, 230) '+p+'%, #f7f7f7 '+p+'%)'
        },
        destroy(){
            // document.querySelector('.pie-wrapper').style.display = 'none'
            document.getElementById('progress').style.background = ''
        }
    }
}


function nextFrame(){
    return new Promise(next => setTimeout(next, 10 ))
}

// -webkit-linear-gradient(left, rgb(204, 208, 230) 50%, #f7f7f7 50%)