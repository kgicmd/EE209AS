//
//  ViewController.swift
//  smallKbrd
//
//  Created by Hongyan Gu on 10/24/18.
//  Copyright © 2018 kgicmd. All rights reserved.
//

import UIKit

var pickerDataSource = ["a", "e", "o", "t"];
var pickerDataSource3 = ["c", "d", "h", "i","l","n","r","s"];
var pickerDataSource2 = ["b","f","g","j","k","m","p","q","u","v","w","x","y","z"];

var show1 = "a"
var show2 = "b"
var show3 = "c"

class ViewController: UIViewController, UIPickerViewDataSource, UIPickerViewDelegate {
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        if(pickerView.tag == 0) {
            return pickerDataSource.count
        }else if (pickerView.tag == 1){
            return pickerDataSource2.count
        }else{
            return pickerDataSource3.count
        }
    }
    
    func pickerView(_ pickerView: UIPickerView, rowHeightForComponent component: Int) -> CGFloat {
        return 30.0
    }
    
    
    
    @IBOutlet weak var textField: UITextView!
    @IBOutlet weak var picView1: UIPickerView!
    @IBOutlet weak var picView2: UIPickerView!
    @IBOutlet weak var picView3: UIPickerView!
    
    @IBOutlet weak var btn1: UIButton!
    @IBOutlet weak var btn2: UIButton!
    @IBOutlet weak var btn3: UIButton!
    
    @IBOutlet weak var enterBtn: UIButton!
    @IBOutlet weak var shiftBtn: UIButton!
    @IBOutlet weak var delBtn: UIButton!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        textField.inputView = UIView()
        // Do any additional setup after loading the view, typically from a nib.
        picView1.delegate = self
        picView2.delegate = self
        picView3.delegate = self
        picView1.dataSource = self
        picView2.dataSource = self
        picView3.dataSource = self
        //picView1.backgroundColor = UIColor(hue: 0/25, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        //picView2.backgroundColor = UIColor(hue: 4/25, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        //picView3.backgroundColor = UIColor(hue: 18/25, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        
    }
    // MARK: UIPickerView Delegation
    
    func pickerView( _ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        if (pickerView.tag == 0){
            return pickerDataSource[row]
        }else if (pickerView.tag == 1) {
            return pickerDataSource2[row]
        }else{
            return pickerDataSource3[row]
        }
    }
    
    func pickerView( _ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        if (pickerView.tag == 0){
            show1 =  pickerDataSource[row]
            //let hue = CGFloat(row)/25
            //pickerView.backgroundColor = UIColor(hue: hue, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        }else if (pickerView.tag == 1) {
            show2 =  pickerDataSource2[row]
            //let hue = (CGFloat(row)+4)/25
            //pickerView.backgroundColor = UIColor(hue: hue, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        }else {
            show3 =  pickerDataSource3[row]
            //let hue = (CGFloat(row)+18)/25
            //pickerView.backgroundColor = UIColor(hue: hue, saturation: 0.5, brightness: 1.0, alpha: 0.5)
        }
    }
    
    
    
    @IBAction func confirm1(_ sender: Any) {
        textField.text = textField.text + show1
    }
    
    @IBAction func confirm2(_ sender: Any) {
        textField.text = textField.text + show2
    }
    
    @IBAction func confirm3(_ sender: Any) {
        textField.text = textField.text + show3
    }
    
    @IBAction func enterANenter(_ sender: Any) {
        textField.text = textField.text + "\n"
    }
    @IBAction func delACh(_ sender: Any) {
        textField.text = String(textField.text.dropLast())
    }
    
    @IBAction func enterASpace(_ sender: Any) {
        textField.text = textField.text + " "
    }
    
}

