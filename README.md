# u-net代码 / pytorch	<br>
使用overlap-tile+镜像策略而不是用padding，解决匹配不一致问题 <br>
使用的代码经过修改有一些问题 <br>
如果发现了代码有什么问题或者提出一些建议可以通过邮箱告诉我：
email：wi.nux@qq.com
## 代码文件
* mian.py <br>
代码运行文件，包含train和test功能
* overlap-tile.py <br>
镜像翻转，图片切片和图片重叠
* data-process.py <br>
数据导入，以及数据处理
