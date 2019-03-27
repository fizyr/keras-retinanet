import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import NavBar from 'antd-mobile/lib/nav-bar';
import Icon from 'antd-mobile/lib/icon';
import ImagePicker from 'antd-mobile/lib/image-picker'
import WingBlank from 'antd-mobile/lib/wing-blank'
import WhiteSpace from 'antd-mobile/lib/white-space'
import Button from 'antd-mobile/lib/button'
import Toast from 'antd-mobile/lib/toast'
import 'antd-mobile/dist/antd-mobile.css'
import Axios from 'axios';

const serverURL = '/detect'

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      fileList: [],
      detectResult: '',
      loading: false
    }
    this.onSelectImage = this.onSelectImage.bind(this)
    this.onUpload = this.onUpload.bind(this)
  }

  onSelectImage(files, type, index) {
    this.setState({ fileList: files })
  }

  async onUpload(){
    if(this.state.fileList.length !== 1){
      Toast.info('请先选择文件')
      return
    }
    this.setState({loading:true})
    let formData = new FormData()
    formData.append('image', this.state.fileList[0].file)
    let res = await Axios.post(serverURL, formData, {headers:{
      'Content-Type':'multipart/form-data'
    }})
    this.setState({detectResult:res.data})
    this.setState({loading:false, fileList:[]})
  }

  render() {
    let Result = this.state.detectResult ? <div>
      <p class="result">检测结果：</p>
      <img class="result-img" src={`${serverURL}?image=${this.state.detectResult}`} />
    </div> : <div></div>
    return (
      <div>
        <NavBar
          mode="dark"
        >Group-8 汽车划痕检测</NavBar>
        <WhiteSpace></WhiteSpace>
        <WingBlank>

              <ImagePicker
                files={this.state.fileList}
                onChange={this.onSelectImage}
                onImageClick={(index, fs) => console.log(index, fs)}
                selectable={this.state.fileList < 1}
                multiple={false}
                length="1"
                accept="image/*"
              />
              <WhiteSpace />
              <Button type="primary" onClick={this.onUpload} loading={this.state.loading} disabled={this.state.loading}>智能识别</Button>
              <WhiteSpace />
              {Result}
        </WingBlank>

      </div>
    );
  }
}

export default App;
