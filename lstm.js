

const LSTM = `
    // Tensor output: [Ns, 1, 2]
    uniform Tensor X; // [Ni, 1, 1]
    uniform Tensor prev; // [Ns, 1, 2]
    uniform Tensor W; // [Ns, Ns + Ni + 1, 4]
    const int Ni = #(X.shape).x;
    const int Ns = #(W.shape).x;

    float tanh(float x){
        float e = exp(2.0 * clamp(x, -10.0, 10.0) );
        return (e-1.0)/(e+1.0);
    }
    float sigmoid(float x){ return 1.0/(1.0+exp(-clamp(x, -10.0, 10.0))); }
    float hard_sigmoid(float x){ return clamp(x * 0.2 + 0.5, 0.0, 1.0); }

    vec4 process4(ivec4 pos) {
        int j = pos.x;
        vec4 fioc = W.read4(j, 0); // bias
        for(int k = 0; k < Ni; k++) // inputs
            fioc += W.read4(j, 1 + k) * X.read4(k, 0).x; 
        for(int k = 0; k < Ns; k++) // prev outputs
            fioc += W.read4(j, 1 + Ni + k) * prev.read4(k, 0).x; 
        float c_t = hard_sigmoid(fioc.x) * prev.read4(j, 0).y 
                  + tanh(fioc.w) * hard_sigmoid(fioc.y); // state
        float h_t = tanh(c_t) * hard_sigmoid(fioc.z); // output
        return vec4(h_t, c_t, 0, 0);
    }
`


const TextureBuffer = `
    uniform Tensor buffer;
    uniform Tensor data;
    uniform int index;

    vec4 process4(ivec4 pos){
        if(pos.w == index){
            return data.read4(ivec4(pos.xyz, 0));
        }else{
            return buffer.read4(pos);
        }
    }
`

const FullyConnected = `
    uniform Tensor inputs;
    uniform Tensor W;
    uniform Tensor b;
    const int length = #(inputs.shape).x;

    vec4 process4(ivec4 pos) {
        vec4 sum = vec4(0, 0, 0, 0);
        for(int i = 0; i < length; i++){
            sum += inputs.read4(i, 0) * W.read4(i, pos.x);
        }
        return sum + b.read4(pos);
    }
`

const HardMax = `
    uniform Tensor data;
    const int length = #(data.shape).x;

    vec4 process4(ivec4 pos) {
        vec4 maxValue = vec4(-10000, -10000, -10000, -10000);
        for(int i = 0; i < length; i++){
            maxValue = max(maxValue, data.read4(i, 0));
        }
        return vec4(greaterThanEqual(data.read4(pos), maxValue));
    }
`

const WarmSample = `
    uniform Tensor data;
    uniform float temperature;
    uniform float random;

    const int length = #(data.shape).x;
    
    vec4 process4(ivec4 pos) {
        float sum = 0.0;
        for(int i = 0; i < length; i++){
            sum += exp(data.read4(i, 0).x / temperature);
        }
        float samp = 0.0;
        for(int i = 0; i < length; i++){
            float range = exp(data.read4(i, 0).x / temperature) / sum;
            if(random > samp && random < samp + range){
                return vec4(i, 0, 0, 0);
            }
            samp += range;
        }
        return vec4(0, 0, 0, 0);
    }
`

const OneHot = `
    uniform Tensor data;
    
    vec4 process4(ivec4 pos) {
        if(abs(float(pos.x) - data.read4(0, 0).x) < 1.0){
            return vec4(1, 1, 1, 1);
        }else{
            return vec4(0, 0, 0, 0);
        }
    }
`
