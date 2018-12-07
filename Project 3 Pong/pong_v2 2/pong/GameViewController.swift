//
//  GameViewController.swift
//  pong
//
//  Created by Jojo Chen on 10/28/18.
//  Copyright © 2018 Jojo Chen. All rights reserved.
//

import UIKit
import SpriteKit
import GameplayKit
import Speech
import GameKit
import AVFoundation
import CoreMotion


class GameViewController: UIViewController , SFSpeechRecognizerDelegate, AVAudioPlayerDelegate{

    var gameScene : GameScene!
    
    
    var motionManager = CMMotionManager()
    //speech transcription
    
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale.init(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let AVA = AVAudioEngine()
    
    
    @IBOutlet weak var countDownLabel: UILabel!
    @IBOutlet weak var micro: UIButton!
    @IBOutlet weak var textView: UITextView!
    
    @IBOutlet weak var playerNum: UIButton!
    @IBOutlet weak var label: UILabel!
    
    override func viewDidAppear(_ animated: Bool) {
//        motionManager.gyroUpdateInterval = 0.01
//        motionManager.startGyroUpdates(to: OperationQueue.current!){ (data, error) in
//            if let myData = data {
//                if myData.rotationRate.z < -0.4 {
////                    print(myData.rotationRate.x)
//                    //print(myData.rotationRate.z)
//                    self.gameScene.moveRight2()
////                    print(myData.rotationRate.z)
//                }else if myData.rotationRate.z > 0.4 {
////                    print(myData.rotationRate.x)
//                    //print(myData.rotationRate.z)
//                    self.gameScene.moveLeft2()
//                }
//
//            }
        
        /*if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.01
            motionManager.startDeviceMotionUpdates(to: OperationQueue.current!) {
                (data, error) in
                if let data = data {
                    let rotation = atan2(data.gravity.x, data.gravity.y)
                    print(rotation)
                    
                }
            }*/
//        }
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    var counter : Int = 10

    override func viewDidLoad() {
        super.viewDidLoad()
        

        //hide text view & micro & label
        textView.isHidden = true
        label.isHidden = true
        
        
        if let view = self.view as! SKView? {
            // Load the SKScene from 'GameScene.sks'
            if let scene = SKScene(fileNamed: "GameScene") {
                // Set the scale mode to scale to fit the window
                scene.scaleMode = .aspectFill
                
                //scene.referenceOfGameViewController = self
                // Present the scene
                
                
                view.presentScene(scene)
                gameScene = scene as? GameScene
                gameScene.referenceOfGameViewController = self
            }
            
            view.ignoresSiblingOrder = true
            
            view.showsFPS = true
            view.showsNodeCount = true
        }
        micro.isEnabled=false
        speechRecognizer.delegate=self
        SFSpeechRecognizer.requestAuthorization{(authStatus) in
            var isButtonEnabled = false
            
            switch authStatus {
            case .authorized:
                isButtonEnabled = true
            case .denied:
                isButtonEnabled = false
            case .restricted:
                isButtonEnabled = false
            case .notDetermined:
                isButtonEnabled = false
            }
            OperationQueue.main.addOperation() {
                self.micro.isEnabled = isButtonEnabled
            }
        }
    }

    func startRecording() {
        
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }
        
        let audio = AVAudioSession.sharedInstance()
        do {
            try audio.setCategory(AVAudioSessionCategoryRecord)
            try audio.setMode(AVAudioSessionModeMeasurement)
            try audio.setActive(true, with: .notifyOthersOnDeactivation)
        } catch {
            print("ERROR!")
        }
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        let inputNode = AVA.inputNode
        
        recognitionRequest?.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest!, resultHandler: { (result, error) in
            
            var isFinal = false
            
            if let result = result {
                let bestString = result.bestTranscription.formattedString
                self.textView.text = bestString
                
                var lastString: String = ""
                for segment in result.bestTranscription.segments {
                    let indexTo = bestString.index(bestString.startIndex, offsetBy: segment.substringRange.location)
                    lastString = bestString.substring(from: indexTo)
                }
                self.checkForColorsSaid(resultString: lastString)
                isFinal = (result.isFinal)
            }
            
            if error != nil || isFinal {
                self.AVA.stop()
                inputNode.removeTap(onBus: 0)
                
                self.recognitionRequest = nil
                self.recognitionTask = nil
                
                self.micro.isEnabled = true
            }
        })
        
        let rec = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 512, format: rec) { (buffer, when) in
            self.recognitionRequest?.append(buffer)
        }
        
        AVA.prepare()
        
        do {
            try AVA.start()
        } catch {
            print("Error!")
        }
        
        textView.text = "I am ready for speech transcription!"
        
    }
    
    
    //discussion switch
    func checkForColorsSaid(resultString: String) {
        
        switch resultString {
        case "blue", "Blue":
            gameScene.main.run(SKAction.moveTo(x: 125.0, duration: 0.5))
            print("right")
        case "Yellow", "yellow":
            gameScene.main.run(SKAction.moveTo(x: 375.0, duration: 0.5))
        case "Green", "green":
            gameScene.main.run(SKAction.moveTo(x: 625.0, duration: 0.5))
            print("left")
        default: break
        }
    }
    
    @IBAction func multiPlayer(_ sender: AnyObject) {
        gameScene.playerOption = 2
        playerNum.isHidden = true
        
    }
    
    @IBAction func tap(_ sender: AnyObject) {
        if AVA.isRunning {
            AVA.stop()
            recognitionRequest?.endAudio()
            micro.isEnabled = false
            micro.setTitle("＊Tap To Start", for: .normal)
        } else {
            startRecording()
            micro.setTitle("I'm Done!", for: .normal)
            micro.isHidden = true
        }
    }
    
}
