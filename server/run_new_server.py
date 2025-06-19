from new_app import app

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible from the network
    app.run(host='0.0.0.0', port=8080, debug=True)
