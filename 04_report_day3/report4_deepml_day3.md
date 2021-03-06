# 第４ステージ　深層学習（後編）day3
* RNNの概要  
  時系列データに対応可能なニューラルネットワークである。  
  * 時系列データとは、  
    * 時間的順序を追って一定間隔ごとに観察される。  
    * 相互に統計的依存関係が認められるデータの系列  
  * 時系列データの例  
    * 音声データ  
    * テキストデータ  
* RNNの特徴  
  時系列モデルを扱うには、初期の状態と過去の時間t-1の状態を保持し、そこから次の時間でのtを再帰的に求める再帰構造が必要になる。  
  * 数学的記述  
    $$ 中間層への総和 :\boldsymbol{u}^t = W_{in}\boldsymbol{x}^t + Wz^{t-1}+b $$
    $$ 中間層の活性化関数：z^t = f( W_{in}\boldsymbol{x}^t + Wz^{t-1}+b) $$
    $$ 出力層への総和 :v^t = W_{out}z^t + C $$
    $$ 出力層の活性化関数 :y^t = g(W_{out}z^t + C) $$  
  このように、重みWについて、入力層から中間層へ出力するときかけあわされる重みW_inと、中間層から出力層へ出力するときにかけあわされる重みW_outと、最後に最もRNNにとって重要な重みである中間層(t-1)から次の中間層(t)へ出力するときにかけあわされるWという３つがある。よって、入力層から中間層への出力u_tは、２つの重みの式となる。  

>【（確認問題）誤差逆伝播法で使用される連鎖律】  
  連鎖律の原理を使い、dz/dxを求めよ。$$ z = t^2$$ $$t=x+y $$
  【回答】  
    $\frac{dz}{dx} = \frac{dz}{dt}\frac{dt}{dx} = 2t=2(x+y) $

---
* BPTT (Back Propagation Through Time)  
  RNN版の誤差逆伝播法。  
  勾配消失や勾配爆発が起きやすい。時系列を遡るほどに中間層が多くなってくるため。  
  勾配爆発を防止するという観点では活性化関数にRelu関数ではなくシグモイド関数やtanh関数などを適用することも手段の１つ。勾配消失の課題解決には後述するLSTMなどがある。  
  以下は、重みとバイアスの勾配の数式。  
  重みやバイアスの更新は、現在の値に以下の勾配に学習率を乗じたもの引き算して更新していく。
    $$ ①\frac{\partial{E}}{\partial{W_{(in)}}} = \frac{\partial{E}}{\partial{u^t}}[\frac{\partial{u^t}}{\partial{W_{(in)}}}]^T=\delta^t[x^t]^T  $$  
    $$ ②\frac{\partial{E}}{\partial{W_{(out)}}} = \frac{\partial{E}}{\partial{v^t}}[\frac{\partial{v^t}}{\partial{W_{(out)}}}]^T=\delta^{out,t}[z^t]^T  $$  
    $$ ③\frac{\partial{E}}{\partial{W}} = \frac{\partial{E}}{\partial{u^t}}[\frac{\partial{u^t}}{\partial{W}}]^T=\delta^t[z^{t-1}]^T  $$  
    $$ ④\frac{\partial{E}}{\partial{b}} = \frac{\partial{E}}{\partial{u^t}}\frac{\partial{u^t}}{\partial{b}}=\delta^t  $$  
    $$ ⑤\frac{\partial{E}}{\partial{c}} = \frac{\partial{E}}{\partial{v^t}}\frac{\partial{v^t}}{\partial{v}}=\delta^{out,t}  $$  
  ソースコードにすると、「サンプルコード3_1_simple_RNN.ipynb」より抜粋。
  ``` python
    for t in range(binary_dim)[::-1]:
        X = np.array([a_bin[-t-1],b_bin[-t-1]]).reshape(1, -1)        

        delta[:,t] = (np.dot(delta[:,t+1].T, W.T) + np.dot(delta_out[:,t].T, W_out.T)) * functions.d_sigmoid(u[:,t+1])

        # 勾配更新
        W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(-1,1))
        W_grad += np.dot(z[:,t].reshape(-1,1), delta[:,t].reshape(1,-1))
        W_in_grad += np.dot(X.T, delta[:,t].reshape(1,-1))
    
    # 勾配適用
    W_in -= learning_rate * W_in_grad
    W_out -= learning_rate * W_out_grad
    W -= learning_rate * W_grad
  ```  
* サンプルコード　3_1_simple_RNN_after_kenfuruki.ipynb  
  中間層の活性化関数につきシグモイド、Relu、tanhでそれぞれ実行。  
  それ以外は何も変更せず。  
  実行したファイルのイテレーションごとの誤差グラフを付けた。  
  「結果_RNN_XXX.png」XXXは各関数名。  
  Reluに関しては勾配更新が０になっている状況で、以降学習が進まず。  
  tanhに関しては、学習が収束したが、シグモイド関数よりは学習に時間がかかった。


---
* LSTM（Long short-term memory）  
  RNNの課題として時系列を遡れば遡るほど、勾配が消失していく問題がある。これをネットワークの構造自体を変えて解決したモデルがLSTMになる。学習を行う層と時系列の記憶を担当する層とに役割分担させたネットワーク構造。これにより長期記憶が実現できる構造となった。ただし課題を克服する過程により複雑化しパラメータが多く計算負荷が高いというのがデメリット。  
    * CEC（Constand Error Carousel）  
      学習と記憶のうち、記憶のみを担当する中間層。RNNのBPTTになると、勾配の値が爆発的に増加か、勾配消失しやすい性質にある。勾配爆発や勾配消失を起こさないために勾配＝１となるよう重みWを調整する。CEC単独だと過去の情報をすべて保持しているため、その情報が必要なくなった後も、影響を及ぼしてしまう。また時間依存度に関係なく、重みが一律になってしまう。このような課題は入力ゲート出力ゲート忘却ゲートなどにより解決させる。  
    * 入力ゲート・出力ゲート  
      入力ゲートは１つ前のユニットの入力をどの程度受け取るか（一律ではなく可変の重み）を調整する。  
      出力ゲートは１つ前のユニットの入出力をどの程度受け取るか（一律ではなく可変の重み）を調整する。  
    * 忘却ゲート  
      CECの中身をどの程度残すか（不必要な情報を忘却する）を調整する。  
    * ゲートの基本的理解  
      ゲートとはデータを通過させる割合を決める機構。その活性化関数には出力が０～１に制約されるシグモイド関数がよく用いられる。  
      ゲートの基本的な構成は、２つの入力と１つの出力があり、入力２はシグモイド関数を通ることで、各要素が０～１となり、これと入力１とのアダマール積が出力となる。この処理により各予想に０～１倍の重みを付け、どの程度通過させるかを制御することになっている。  
      > 【（確認問題）LSTMのゲート】  
      以下の文章をLSTMに入力し、空欄に当てはまる単語を予測したいとする。「映画おもしろかったね。ところで、とてもお腹が空いたから何か____。」文中の「とても」という言葉は空欄の予測においてなくなっても影響を及ぼさないと考えられる。このような場合、どのゲートが作用すると考えられるか。  
      ＜回答＞  
      忘却ゲート  
  
    * クリッピング  
      勾配のノルムが閾値を超えた場合、勾配のノルムを正規化することで勾配爆発を防ぐ手法。  
      以下実装。  
      ``` python  
      def gradient_clipping(grad, threshould):
        norm = np.linalg.norm(grad)
        rate = threshould / norm

        if rate < 1:
          return grad * rate
        return grad
      ```  
---
* GRU（Gated Recurrent Unit）  
  従来のLSTMではパラメータが多数存在していたため、計算負荷が高かった。GRUではパラメータ数を大幅に削減し、精度は同等以上になった。  
  以下の２つのゲートで構成される。  
    * リセットゲート  
      過去の隠れ状態を弱める程度を決めるゲート
    * 更新ゲート  
      過去の隠れ状態と仮の隠れ状態の混合割合を決めるゲート  
  
---  
* 双方向RNN（Bidirectional RNN）  
  過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデル。文章の推敲や機械翻訳に使用される。例えば文章内の文字の穴埋めタスクなど、過去から現在だけでなく、未来から現在までの系列情報を用いることが有効と考えられるタスクに用いられる。  
    
---  
* Seq2Seq  
  通常のRNNは入力と出力の長さや順序が同じでなければならない。  
  一方、Seq2SeqはEncoder-Decoderモデルの一種で、入力側と出力側で別々のRNNを使う。機械対話や機械翻訳に使用される。  
  * Encoder RNN  
    ユーザーがインプットしたテキストデータを単語等のトークンに区切って渡す構造。vec1をRNNに入力し、hidden stateを出力。このhidden stateと次の入力vec2をまたRNNに入力しhidden stateを出力する。最後のvecを入れたときのhidden stateをfinal stateをしてとっておく。このfinal stateがthought vectorと呼ばれ、入力した文の意味を表すベクトルとなる。  
  * Decoder RNN  
    システムがアウトプットデータを単語等のトークンごとに生成する構造。  
  
---  
* Word2vec  
  RNNでは、単語のような可変長の文字列をNNに与えることはできない。word2vecでは学習データからボキャブラリを作成し、「ボキャブラリ数×任意の単語ベクトルの次元数」の重み行列により、大規模データの分散表現の学習を現実的な計算速度とメモリ量で実現可能にした。結果の数値ベクトルは、生のテキストをデータの視覚化、機械学習、および深層学習に適した数値表現に変換するために使用できる。  
  
---  
* Attention Mechanism  
  RNNでは、単語のような可変長の文字列をNNに与えることはできない。word2vecでは学習データからボキャブラリを作成し、「ボキャブラリ数×任意の単語ベクトルの次元数」の重み行列により、大規模データの分散表現の学習を現実的な計算速度とメモリ量で実現可能にした。結果の数値ベクトルは、生のテキストをデータの視覚化、機械学習、および深層学習に適した数値表現に変換するために使用できる。