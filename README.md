## WEB interface
```
curl 'http://localhost:3005/api/getresult' \
  -H 'Connection: keep-alive' \
  -H 'Pragma: no-cache' \
  -H 'Cache-Control: no-cache' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36' \
  -H 'Content-Type: text/plain;charset=UTF-8' \
  -H 'Accept: */*' \
  -H 'Origin: https://code2vec.org' \
  -H 'Sec-Fetch-Site: cross-site' \
  -H 'Sec-Fetch-Mode: cors' \
  -H 'Sec-Fetch-Dest: empty' \
  -H 'Referer: https://code2vec.org/' \
  -H 'Accept-Language: zh-CN,zh;q=0.9' \
  --data-binary $'void f(HttpPost method, HttpReadResult result) throws ConnectionException {\n    HttpClient client = HttpConnectionApacheCommon.getHttpClient(data.getSslMode());\n    method.setHeader("User-Agent", HttpConnection.USER_AGENT);\n    if (getCredentialsPresent()) {\n        method.addHeader("Authorization", "Basic " + getCredentials());\n    }\n    HttpResponse httpResponse = client.execute(method);\n    StatusLine statusLine = httpResponse.getStatusLine();\n    result.statusLine = statusLine.toString();\n    result.setStatusCode(statusLine.getStatusCode());\n    result.strResponse = HttpConnectionApacheCommon.readHttpResponseToString(httpResponse);\n}' \
  --compressed
```  


```python
python3
>>> from gensim.models import KeyedVectors as word2vec
>>> vectors_text_path = 'models/java_model/targets.txt'
>>> model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
>>> model.most_similar(positive=['equals', 'tolower'])
>>> model.most_similar(positive=['download', 'send'], negative=['receive'])
```
The above python commands will result in the closest name to both "equals" and "tolower", which is "equalsignorecase".
