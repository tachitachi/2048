from flask_socketio import SocketIO, emit
from flask import Flask, render_template, send_from_directory, request
import numpy as np
from model import Model, ReplayBuffer
import time

app = Flask(__name__)
socketio = SocketIO(app)

obs_space = (4, 4, 16)
action_space = 4
buf = ReplayBuffer(100000)
model = Model(obs_space, action_space)
model.init()
epsilon = 1
decay = 0.0001
prev = {'state': None, 'count': 0}

def to_state(data, key='state'):
    size = data[key]['grid']['size']
    assert(size == obs_space[0] and size == obs_space[1])
    x = np.zeros(obs_space, dtype=np.int32)
    for i in range(size):
        for j in range(size):
            val = data[key]['grid']['cells'][i][j]
            if val is not None:
                x[i][j][np.log2(val['value']).astype(np.int32) - 1] = 1
    return x

@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/meta/<path:path>')
def send_meta(path):
    return send_from_directory('meta', path)

@app.route('/style/<path:path>')
def send_style(path):
    return send_from_directory('style', path)

@socketio.on('connect', namespace='/')
def connect():
    print("connect ", request.sid)
    emit('reset');

@socketio.on('chat message', namespace='/')
def message(data):
    print("message ", data)
    emit('reply', room=request.sid)

@socketio.on('disconnect', namespace='/')
def disconnect():
    print('disconnect ', request.sid)

@socketio.on('state', namespace='/')
def state(data):
    global epsilon
    global prev
    #print("state ", data)
    
    x0 = to_state(data, 'state0')
    action = data['action']
    reward = data['reward']
    x1 = to_state(data, 'state1')


    #print(x, data['action'], data['state1']['score'], data['state1']['over'])
    value = data['state1']['score']
    buf.add((x0, action, reward, x1))

    if prev['state'] is not None:
        if np.sum(prev['state'] - x1) == 0:
            if prev['count'] > 20:
                emit('reset')
                return
            else:
                prev['count'] += 1
        else:
            prev['state'] = x1
            prev['count'] = 0
    else:
        prev['state'] = x1
        prev['count'] = 0

    if len(buf) > 128:
        sample_x, sample_a, sample_r, sample_x1 = buf.sample(128)
        loss = model.train(sample_x, sample_a, sample_r, sample_x1)
        print(loss, epsilon)
    
    if not data['state1']['over']:
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = int(model.predict([x0])[0])
        emit('move', {'dir': action})
        epsilon -= decay
        epsilon = max(epsilon, decay * 10)
    else:
        emit('reset')


@socketio.on('on reset', namespace='/')
def on_reset(data):
    global epsilon
    x = to_state(data, 'state0')
    if np.random.random() < epsilon:
        action = np.random.randint(4)
    else:
        action = int(model.predict([x])[0])
    emit('move', {'dir': action})

if __name__ == '__main__':
    print('listening')
    socketio.run(app, host="0.0.0.0", port=8000)