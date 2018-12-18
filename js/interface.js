

function Connection(){


	var socket = io();
	this.socket = socket;

	socket.on('connect', function(){

		console.log(socket, window.game);

		socket.on('disconnect', function(){

		});

		socket.on('move', function(data){
			let state0 = window.game.serialize();
			window.game.move(data.dir);
			let state1 = window.game.serialize();
			let reward = state1['score'] - state0['score']
			socket.emit('state', {'state0': state0, 'state1': state1, 'action': data.dir, 'reward': reward});
		});

		socket.on('reset', function(data){
			window.game.restart();
			let state0 = window.game.serialize();
			socket.emit('on reset', {'state0': state0});
		});

	});

}

window.requestAnimationFrame(function () {
	console.log('on connect?');
	window.connection = new Connection();
});