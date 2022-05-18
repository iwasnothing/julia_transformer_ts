using Flux
using Transformers
using Transformers.Basic 
using TimeSeries
using Flux: gradient
using Flux.Optimise: update!



function get_src_trg(sequence, enc_seq_len, dec_seq_len, target_seq_len)
	nseq = size(sequence)[2]
	
	@assert  nseq == enc_seq_len + target_seq_len
	src = sequence[:,1:enc_seq_len,:]
	trg = sequence[:,enc_seq_len:nseq-1,:]
	@assert size(trg)[2] == target_seq_len
	trg_y = sequence[:,nseq-target_seq_len+1:nseq,:]
	@assert size(trg_y)[2] == target_seq_len
	if size(trg_y)[1] == 1
	 	return src, trg, dropdims(trg_y; dims=1)
	else
		return src, trg, trg_y
	end
end

function generate_seq(x, seq_len)
	result = Matrix{Float64}[]
	for i in 1:length(x)-seq_len+1
		ele = reshape(x[i:i+seq_len-1],(seq_len,1))	
		push!(result,ele)
	end
	return result
end

begin
	## Model parameters
	dim_val = 512 # This can be any value. 512 is used in the original transformer paper.
	n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
	n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
	n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
	input_size = 1 # The number of input variables. 1 if univariate forecasting.
	dec_seq_len = 92 # length of input given to decoder. Can have any integer value.
	enc_seq_len = 153 # length of input given to encoder. Can have any integer value.
	output_sequence_length = 58 # Length of the target sequence, i.e. how many time steps should your forecast cover
	in_features_encoder_linear_layer = 2048 # As seen in Figure 1, each encoder layer has a feed forward layer. This variable determines the number of neurons in the linear layer inside the encoder layers
	in_features_decoder_linear_layer = 2048 # Same as above but for decoder
	max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder


	#define 2 layer of transformer
	encode_t1 = Transformer(dim_val, n_heads, 64, 2048;future=false,pdrop=0.2)
	
	#define 2 layer of transformer decoder
	decode_t1 = TransformerDecoder(dim_val, n_heads, 64, 2048,pdrop=0.2)

	encoder_input_layer = Dense(input_size,dim_val)
	decoder_input_layer = Dense(input_size,dim_val)
	positional_encoding_layer = PositionEmbedding(dim_val)
	p = 0.2
	dropout_encoder = Dropout(p)
	dropout_decoder = Dropout(p)
	dropout_pos_enc = Dropout(p)
	
	#define the layer to get the final output probabilities
	#linear = Positionwise(Dense(dim_val, output_sequence_length))
	linear = Dense(output_sequence_length*dim_val,output_sequence_length)
	function encoder_forward(x)
	  x = encoder_input_layer(x)
	  e = positional_encoding_layer(x)
	  t1 = x .+ e
	  e = dropout_pos_enc(t1)
	  t1 = encode_t1(t1)
	  t1 = dropout_encoder(t1)
	  return t1
	end
	
	function decoder_forward(tgt, t1)
	  decoder_output = decoder_input_layer(tgt)
	  t2 = decode_t1(decoder_output,t1)
	  t2 = dropout_decoder(t2)
	  t2 = Flux.flatten(t2)
	  p = linear(t2)
	  
	  return p
	end
	function loss(src, trg, trg_y)
	  enc = encoder_forward(src)
	  dec = decoder_forward(trg, enc)
	  err = Flux.Losses.mse(dec,trg_y)
	  return err
	end


	ta = readtimearray("rate.csv", format="mm/dd/yy", delim=',')
	data = generate_seq(values(ta[:"10 YR"]),enc_seq_len+output_sequence_length)
	data = reduce(hcat,data)
	data = convert(Array{Float32,2}, data)
	ps = params(encoder_input_layer, positional_encoding_layer, encode_t1, decoder_input_layer, decode_t1,  linear)
	opt = ADAM(1e-4)
	train_loader = Flux.Data.DataLoader(data, batchsize=32) 
	for i = 1:1000
		for x in train_loader
			sz = size(x)
			sub_sequence = reshape(x,(1,sz[1],sz[2]))
		   	src, trg, trg_y = get_src_trg(
							    sub_sequence,
							    enc_seq_len,
							    dec_seq_len,
							    output_sequence_length
							    )
			grad = gradient(()->loss(src, trg, trg_y), ps)
		    if i % 10 == 0
		        l = loss(x, y)
		    	println("loss = $l")
		    end
		    Flux.update!(opt, ps, grad)
		end
	end
end


















