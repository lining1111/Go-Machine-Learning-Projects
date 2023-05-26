//go:build sidenote
// +build sidenote

package main

import (
	"fmt"
	"strings"
)

func main() {
	a := "The child was learning a new word and was using it excessively. \"shan't!\", she cried"
	dict := make(map[string]struct{})
	words := strings.Fields(a) //words :=strings.Split(a," ")

	for _, word := range words {
		fmt.Println(word)
		dict[word] = struct{}{}
	}
	//fmt.Println(dict)
}
