docker container stop $(docker container ls -aq)
docker container rm $(docker container ls -aq)
docker system prune
docker build -t eegapp:latest .
docker run -d -p 5000:5000 eegapp:latest
start 'chrome.exe' 'http://localhost:5000'