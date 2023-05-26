package main

import (
	"math"
	"math/rand"
	"sync"

	"github.com/chewxy/lingo/corpus"
	"github.com/go-nlp/tfidf"
)

const tiny = 0.0000001

type Class byte //枚举值

const (
	Ham Class = iota
	Spam
	MAXCLASS
)

func (c Class) String() string {
	switch c {
	case Ham:
		return "Ham"
	case Spam:
		return "Spam"
	default:
		panic("HELP")
	}
}

// Example 表示分类样本的结构体
type Example struct {
	Document []string //文件的单词集合
	Class
}

type doc []int

//IDs 此方法是为了适应TFIDF.Add函数入参的接口
func (d doc) IDs() []int { return []int(d) }

type Classifier struct {
	corpus *corpus.Corpus //将单词映射成一个整数，节省内存

	tfidfs [MAXCLASS]*tfidf.TFIDF //保存有关TFIDF的相关状态信息的结构
	totals [MAXCLASS]float64

	ready bool
	sync.Mutex
}

func New() *Classifier {
	var tfidfs [MAXCLASS]*tfidf.TFIDF
	for i := Ham; i < MAXCLASS; i++ {
		tfidfs[i] = tfidf.New()
	}
	return &Classifier{
		corpus: corpus.New(),
		tfidfs: tfidfs,
	}
}

func (c *Classifier) Train(examples []Example) {
	for _, ex := range examples {
		c.trainOne(ex)
	}
}

//trainOne 将Example Document中的每个单词添加到语料库，并获取其ID，将ID添加到doc数据类型，然后将doc添加到TFIDF中。总数加1
func (c *Classifier) trainOne(example Example) {
	d := make(doc, len(example.Document))
	for i, word := range example.Document {
		id := c.corpus.Add(word)
		d[i] = id
	}
	c.tfidfs[example.Class].Add(d)
	c.totals[example.Class]++
}

//后处理
func (c *Classifier) Postprocess() {
	c.Lock()
	if c.ready {
		return
	}

	var docs int
	for _, t := range c.tfidfs {
		docs += t.Docs
	}
	for _, t := range c.tfidfs {
		t.Docs = docs //t.Docs是所有文档总和的docs
		//计算每个词条想对于文档的重要性，这里用的是基于对数的IDF简单计算，当然也可以采用其他计算方式
		//t.CalculateIDF()
		for k, v := range t.TF {
			t.IDF[k] = math.Log1p(float64(t.Docs) / v)
		}
	}
	c.ready = true
	c.Unlock()
}

func (c *Classifier) Score(sentence []string) (scores [MAXCLASS]float64) {
	//得分器是否准备好了
	if !c.ready {
		c.Postprocess()
	}

	d := make(doc, len(sentence))
	for i, word := range sentence {
		id := c.corpus.Add(word)
		d[i] = id
	}
	//计算先验概率
	priors := c.priors()

	// 每个类的得分
	for i := range c.tfidfs {
		score := math.Log(priors[i])
		// 似然性
		for _, word := range sentence {
			prob := c.prob(word, Class(i))
			score += math.Log(prob)
		}

		scores[i] = score
	}
	return
}

//先验概率
func (c *Classifier) priors() (priors []float64) {
	priors = make([]float64, MAXCLASS)
	var sum float64
	for i, total := range c.totals {
		priors[i] = total
		sum += total
	}
	//计算概率分布，即个体数量除以总数
	for i := Ham; i < MAXCLASS; i++ {
		priors[int(i)] /= sum
	}
	return
}

//似然性
func (c *Classifier) prob(word string, class Class) float64 {
	//首先检验该单词，如果单词未出现则给一个极小的默认值，避免除0错误出现
	id, ok := c.corpus.Id(word)
	if !ok {
		return tiny
	}

	freq := c.tfidfs[class].TF[id] //单词在此类中存在的概率
	idf := c.tfidfs[class].IDF[id] //单词对于文档的重要性
	// idf := 1.0

	// a word may not appear at all in a class.
	if freq == 0 {
		return tiny
	}

	return freq * idf / c.totals[class]
}

func (c *Classifier) Predict(sentence []string) Class {
	scores := c.Score(sentence)
	return argmax(scores)
}

func (c *Classifier) unseens(sentence []string) (retVal int) {
	for _, word := range sentence {
		if _, ok := c.corpus.Id(word); !ok {
			retVal++
		}
	}
	return
}

func argmax(a [MAXCLASS]float64) Class {
	max := math.Inf(-1)
	var maxClass Class
	for i := Ham; i < MAXCLASS; i++ {
		score := a[i]
		if score > max {
			maxClass = i
			max = score
		}
	}
	return maxClass
}

func shuffle(a []Example) {
	// r := rand.New(rand.NewSource(time.Now().Unix()))

	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
		// b[i], b[j] = b[j], b[i]
	}
	// for len(a) > 0 {
	// 	n := len(a)
	// 	randIndex := r.Intn(n)
	// 	a[n-1], a[randIndex] = a[randIndex], a[n-1]
	// 	a = a[:n-1]
	// }
}
