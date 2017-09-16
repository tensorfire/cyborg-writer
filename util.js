

async function loadArrayFromURL(fileName){
    var xhr = new XMLHttpRequest()
    xhr.open('GET', fileName, true)
    xhr.responseType = 'arraybuffer'
    xhr.send(null)
    await new Promise(resolve => xhr.onload = resolve)
    var buffer = new Float32Array(xhr.response)
    var shape = fileName.match(/\d+(x\d+)*$/)[0].split('x').map(k => +k)
    return ndarray(buffer, shape)
}


function nextFrame(){
    return new Promise(next => setTimeout(next, 10 ))
}
