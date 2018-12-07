//
//  GameScene.swift
//  pong
//
//  Created by Jojo Chen on 10/28/18.
//  Copyright © 2018 Jojo Chen. All rights reserved.
//

import SpriteKit
import GameplayKit
import Darwin
import CoreMotion

struct BitMask {
    static let Ball: UInt32 = 0x1 << 0
    static let enemy: UInt32 = 0x1 << 1
    static let main: UInt32 = 0x1 << 1
    static let ball: UInt32 = 0x1 << 1
}

class GameScene: SKScene, SKPhysicsContactDelegate {
    
    var ball = SKSpriteNode()
    var enemy = SKSpriteNode()
    var main = SKSpriteNode()
    var ball2 = Ball()
    var extraBall = SKSpriteNode()
    var sun = SKSpriteNode()
    var ballTwo = SKSpriteNode()
    
    var mainScore : SKLabelNode!
    var enemyScore : SKLabelNode!
    var winLogo : SKLabelNode!
    
    weak var referenceOfGameViewController : GameViewController!
    let motionManager = CMMotionManager()
    
    override func didMove(to view: SKView) {
        //gyro
        motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
        motionManager.startDeviceMotionUpdates()
        
        
        ball = self.childNode(withName: "ball") as! SKSpriteNode
        enemy = self.childNode(withName: "enemy") as! SKSpriteNode
        main = self.childNode(withName: "main") as! SKSpriteNode
        extraBall = self.childNode(withName: "extraBall") as! SKSpriteNode
        sun = self.childNode(withName: "sun") as! SKSpriteNode
        ballTwo = self.childNode(withName: "ballTwo") as! SKSpriteNode
        
//        ball.physicsBody?.velocity = (CGVector(dx: 20, dy: 20))
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                self.ball.physicsBody?.applyImpulse(CGVector(dx: 10, dy: 10))
            }

        
//        extraBall.physicsBody?.applyImpulse(CGVector(dx: 10, dy: 10))
//        ballTwo.physicsBody?.applyImpulse(CGVector(dx: 10, dy: 10))
        let border = SKPhysicsBody(edgeLoopFrom: self.frame)
        
        border.friction = 0
        border.restitution = 1
        
        physicsWorld.gravity = CGVector(dx: 0.0, dy: 0.0)
//        physicsWorld.contactDelegate = self
        self.physicsBody = border
        
        //create score labels
        mainScore = SKLabelNode(fontNamed: "Chalkduster")
        mainScore.zPosition = 1
        mainScore.position = CGPoint(x: frame.size.width / 2, y: frame.minY + 40  )
        mainScore.fontSize = 50
        mainScore.text = "0"
        mainScore.fontColor = SKColor.red
        self.addChild(mainScore)
        enemyScore = SKLabelNode(fontNamed: "Chalkduster")
        enemyScore.zPosition = 1
        enemyScore.position = CGPoint(x: frame.size.width / 2, y: frame.maxY - 60.0  )
        enemyScore.fontSize = 50
        enemyScore.text = "0"
        enemyScore.fontColor = SKColor.cyan
        self.addChild(enemyScore)
        //create Win Logo
        winLogo = SKLabelNode(fontNamed: "Chalkduster")
        winLogo.zPosition = 1
        winLogo.position = CGPoint(x:frame.size.width * 2, y:frame.size.height / 2)
        winLogo.fontSize = 100
        self.addChild(winLogo)

        //create extraBall
        
        
        
//        extraBall.isHidden = true
        
 
        
        //create contact
//        extraBall.physicsBody = SKPhysicsBody(circleOfRadius: ball2.radius) // FORGOT THIS
//        extraBall.physicsBody?.categoryBitMask = BitMask.Ball
//        extraBall.physicsBody?.contactTestBitMask = BitMask.ball
//
//        ball.physicsBody = SKPhysicsBody(rectangleOf: ball.size) // FORGOT THIS
//        ball.physicsBody?.categoryBitMask = BitMask.ball
//        ball.physicsBody?.contactTestBitMask = BitMask.Ball
//
////        ball2.physicsBody?.affectedByGravity = false
//        ball.physicsBody?.allowsRotation = false
//        ball.physicsBody?.affectedByGravity = false
//        ball.physicsBody?.linearDamping = 0
//        ball.physicsBody?.angularDamping = 0
//        ball.physicsBody?.friction = 0
//        ball.physicsBody?.applyImpulse(CGVector(dx: 100, dy: 100))
//
//

//        //ball
//        ball2.position = CGPoint(x: frame.size.width / 2, y: frame.size.height / 2)
//        var a = CGFloat()
//        let b = Double( (arc4random_uniform(10)) / 10)
//
//        if b < 0.5{
//            a = CGFloat(Double( (arc4random_uniform(90) + 45) ))
//
//        }else{
//            a = CGFloat( -1 * Double( (arc4random_uniform(90) + 45) ))
//
//        }
//        let v : CGFloat = 400
//        ball2.physicsBody?.velocity = CGVector(dx: v*cos(a), dy: v*sin(a))
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            self.sun.position = CGPoint(x: 186.0 , y: 508.0)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 20) {
            self.extraBall.position = CGPoint(x: 618.0 , y: 778.0)
        }
    }
    
//    func didBeginContact(contact: SKPhysicsContact) {
//        var firstBody: SKPhysicsBody
//        var secondBody: SKPhysicsBody
//
//        if contact.bodyA.categoryBitMask < contact.bodyB.categoryBitMask {
//            firstBody = contact.bodyA
//            secondBody = contact.bodyB
//        } else {
//            firstBody = contact.bodyB
//            secondBody = contact.bodyA
//        }
//
//        if firstBody.categoryBitMask == BitMask.ball && secondBody.categoryBitMask == BitMask.ball {
//            ball2.isHidden = true
//        }
//    }
    
    
   //movement
    func moveRight(){
        let oneStep = frame.size.width / 15
        
//        let min_pos = min(main.position.x + oneStep, frame.maxX - 170)
        if (frame.maxX - main.position.x ) > 0{
            main.run(SKAction.moveTo(x: main.position.x + oneStep , duration: 0.1))
            
        }else{
            main.run(SKAction.moveTo(x: frame.maxX, duration: 0.1))
        }
//        main.run(SKAction.moveTo(x: min_pos , duration: 0.1))
        
    }
    func moveLeft(){
        let oneStep = frame.size.width / 15
        if ( main.position.x - frame.minX) > 0{
            main.run(SKAction.moveTo(x: main.position.x - oneStep , duration: 0.1))
            
        }else{
            main.run(SKAction.moveTo(x: frame.minX, duration: 0.1))
        }
//        let max_pos = max(main.position.x - oneStep, frame.maxX + 170)
//        main.run(SKAction.moveTo(x: max_pos, duration: 0.1))
        
    }
    //movement2
    func moveRight2(){
        let oneStep = frame.size.width / 15
        if (frame.maxX - enemy.position.x ) > 0{
            enemy.run(SKAction.moveTo(x: enemy.position.x + oneStep , duration: 0.1))
            
        }else{
            enemy.run(SKAction.moveTo(x: frame.maxX, duration: 0.1))
        }
        
    }
    func moveLeft2(){
        let oneStep = frame.size.width / 15
        if ( enemy.position.x - frame.minX) > 0{
            enemy.run(SKAction.moveTo(x: enemy.position.x - oneStep, duration: 0.1))
        }else{
            enemy.run(SKAction.moveTo(x: frame.minX+120, duration: 0.1))
        }
        
    }
    
    // match movement
    /*
    func matchMovement(angle){
        
        var location =
        main.run(SKAction.moveTo(x: main.position.x - oneStep, duration: 0.1))
        
    }*/

    //finger control method
//    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
//        for touch in touches{
//            let location = touch.location(in: self)
//
//            main.run(SKAction.moveTo(x: location.x, duration: 0.2))
//        }
//    }
//
//    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
//        for touch in touches {
//            let location = touch.location(in: self)
//
//            main.run(SKAction.moveTo(x: location.x, duration: 0.2))
//        }
//    }
    func resetBall(){
        ballTwo.position = CGPoint(x: frame.size.width / 2, y: frame.size.height / 2)
        ballTwo.physicsBody?.applyImpulse(CGVector(dx: 10, dy:10))
//        ballTwo.physicsBody?.velocity.dx = 200
//        ballTwo.physicsBody?.velocity.dy = 200
    }
    //restart the game
    func resetGame1(){
        ball.position = CGPoint(x: (frame.size.width) / 2, y: (frame.size.height) / 2 )
        ball.physicsBody?.velocity.dy *= -1
        
        ball.size.width = 50.0
        ball.size.height = 50.0
        
        ball.physicsBody?.velocity.dx = 200
//        ball.physicsBody?.applyImpulse(CGVector(dx:4, dy:4))
//        ball2.position = CGPoint(x: frame.size.width / 2, y: frame.size.height / 2)
    }
    func resetGame(){
//        ball.position = CGPoint(x: (frame.size.width) / 2, y: (frame.size.height) / 2 )
        ballTwo.position = CGPoint(x: frame.size.width / 2, y: frame.size.height / 2)
        ballTwo.physicsBody?.velocity.dy *= -1
    }
    //game score count
    override func didSimulatePhysics() {
            //sun
        if sun.position.x > 187.0 || sun.position.x < 185.0 {
            sun.position = CGPoint(x: 186.0, y: frame.size.height * 2)
            sun.physicsBody?.velocity.dx = 0
            sun.physicsBody?.velocity.dy = 0
            ball.physicsBody?.velocity.dx *= 1
            ball.physicsBody?.velocity.dy *= 1
            ball.size.width = 80.0
            ball.size.height = 80.0
            
        }
        //moon
        if extraBall.position.x > 619 || extraBall.position.x < 617 {
            extraBall.position = CGPoint(x: 618.0, y: frame.size.height * 2)
            extraBall.physicsBody?.velocity.dx = 0
            extraBall.physicsBody?.velocity.dy = 0
            //            ball.physicsBody?.applyImpulse(CGVector(dx: 10, dy: 10))
            ballTwo.physicsBody?.applyImpulse(CGVector(dx: 5, dy: 5))
            
            ball.physicsBody?.velocity.dx *= 3
            ball.physicsBody?.velocity.dy *= 3
            resetBall()
        }
        //earth
        if ball.position.y <= frame.minY + 40 {
            var enemyScoreInt : Int = Int(enemyScore.text!)!
            enemyScoreInt += 1
            enemyScore.text = String(enemyScoreInt)
            resetGame1()
        }else if ball.position.y >= frame.maxY - 40 {
            var mainScoreInt : Int = Int(mainScore.text!)!
            mainScoreInt += 1
            mainScore.text = String(mainScoreInt)
            resetGame1()
        }else if ball.position.x <= frame.minX + 10 || ball.position.x >= frame.size.width - 10{
            ball.physicsBody?.velocity.dx = ball.physicsBody!.velocity.dx * -10 + 10
        }
        //        else if ball.position.y == main.position.y + 100 || ball.position.y == enemy.position.y - 100{
        //            ball.physicsBody?.velocity.dy = ball.physicsBody!.velocity.dx * -1000 + 10
        //            ball.physicsBody?.velocity.dx = 5
        //        }
        //set ball2 boundary
        if ballTwo.position.x < frame.minX + 10 || ballTwo.position.x > frame.size.width - 10 {
            ballTwo.physicsBody?.velocity.dx = ballTwo.physicsBody!.velocity.dx * -1 + 5
        }else if ballTwo.position.y <= frame.minY + 40{
            var enemyScoreInt : Int = Int(enemyScore.text!)!
            enemyScoreInt += 1
            enemyScore.text = String(enemyScoreInt)
            resetGame()
        }else if ballTwo.position.y >= frame.maxY - 40 {
            var mainScoreInt : Int = Int(mainScore.text!)!
            mainScoreInt += 1
            mainScore.text = String(mainScoreInt)
            resetGame()
        }
            //        else if ballTwo.position.y == main.position.y + 100 || ballTwo.position.y == enemy.position.y - 100{
            //            ballTwo.physicsBody?.velocity.dy = ballTwo.physicsBody!.velocity.dx * -1000 + 10
            //            ballTwo.physicsBody?.velocity.dx = 5
            //        }
        
        
    }
    // ball bounce with an angle
    
    func didBegin(_ contact: SKPhysicsContact) {
        // bounce with angle part
        let minAngle = CGFloat.pi/6
        let maxAngle = CGFloat.pi*5/6
        let w = main.frame.width
        let rightOfPaddle = main.position.x + w/2
        let dist = rightOfPaddle - ball.position.x
        
        let angle = minAngle + (maxAngle - minAngle)*(dist/w)
        ball.physicsBody?.velocity.dx = cos(angle)
        ball.physicsBody?.velocity.dy = sin(angle)
//        ball.physicsBody?.applyImpulse(CGVector(dx: 2, dy: -2))
//        ballTwo.physicsBody?.applyImpulse(CGVector(dx:2,dy:-2))
        
    }
    
    //endGame
    func endGame(){
        ball.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: ball.position.y), duration: 0.5)){
            self.ball.isHidden=true
        }
        ballTwo.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: ballTwo.position.y), duration: 0.5)){
            self.ballTwo.isHidden = true
        }
        main.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: main.position.y), duration: 0.5)){
            self.main.isHidden = true
        }
        enemy.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: enemy.position.y), duration: 0.5)){
            self.enemy.isHidden = true
        }
        extraBall.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: extraBall.position.y), duration: 0.5)){
            self.extraBall.isHidden = true
        }
        sun.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: sun.position.y), duration: 0.5)){
            self.sun.isHidden = true
        }
        mainScore.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: mainScore.position.y), duration: 0.5)){
            self.mainScore.isHidden = true
        }
        enemyScore.run(SKAction.move(to: CGPoint(x: frame.size.width * 2, y: enemyScore.position.y), duration: 0.5)){
            self.enemyScore.isHidden = true
        }
    }
    
    var playerOption : Int = 1
    
    override func update(_ currentTime: TimeInterval) {
        // Called before each frame is rendered
//        enemy.run(SKAction.moveTo(x: ball.position.x , duration: 0.5))
//        var playerOption : Int = 1
        var mainScoreInt : Int = Int(mainScore.text!)!
        var enemyScoreInt : Int = Int(enemyScore.text!)!
        if playerOption == 1 {
            let ballPosition = ball.position.y - main.position.y
            let extraBallPosition = ballTwo.position.y - main.position.y
            if ballPosition < extraBallPosition {
                main.run(SKAction.moveTo(x: ball.position.x, duration: 0.5))
            } else if ballPosition > extraBallPosition {
                main.run(SKAction.moveTo(x: ballTwo.position.x, duration: 0.5))
            }
            main.size.width = 170
        } else if playerOption == 2{
            main.size.width = 250
            DispatchQueue.main.asyncAfter(deadline: .now() + 60) {
                if mainScoreInt > enemyScoreInt {
                    self.winLogo.text = "Red Win"
                    self.winLogo.fontColor = SKColor.red
                    self.winLogo.run(SKAction.move(to: CGPoint(x: self.frame.size.width / 2, y: self.frame.size.height / 2), duration: 0.5))
                    self.endGame()
                } else if mainScoreInt < enemyScoreInt {
                    self.winLogo.text = "Blue Win"
                    self.winLogo.fontColor = SKColor.cyan
                    self.winLogo.run(SKAction.move(to: CGPoint(x: self.frame.size.width / 2, y: self.frame.size.height / 2), duration: 0.5))
                    self.endGame()
                }else if mainScoreInt == enemyScoreInt{
                    self.winLogo.text = "Finished"
                    self.winLogo.fontColor = SKColor.yellow
                    self.winLogo.run(SKAction.move(to: CGPoint(x: self.frame.size.width / 2, y: self.frame.size.height / 2), duration: 0.5))
                    self.endGame()
                }else{
                    self.winLogo.text = "Finished"
                    self.winLogo.fontColor = SKColor.yellow
                    self.winLogo.run(SKAction.move(to: CGPoint(x: self.frame.size.width / 2, y: self.frame.size.height / 2), duration: 0.5))
                    self.endGame()
                }
            }
        }
        
        if mainScoreInt > 9 {
            winLogo.text = "Red Win"
            winLogo.fontColor = SKColor.red
            winLogo.run(SKAction.move(to: CGPoint(x: frame.size.width / 2, y: frame.size.height / 2), duration: 0.5))
            endGame()
        }else if enemyScoreInt > 9{
            winLogo.text = "Blue Win"
            winLogo.fontColor = SKColor.cyan
            winLogo.run(SKAction.move(to: CGPoint(x: frame.size.width / 2, y: frame.size.height / 2), duration: 0.5))
            endGame()
        }
        
        //gyro
        if let attitude = motionManager.deviceMotion?.attitude {
//            print(attitude)
            let y = CGFloat(-attitude.pitch * 2 / M_PI)
            let x = CGFloat(-attitude.roll * 2 / M_PI)
            //            print(y)
//            print(x)
            
            if x > 0 {
                moveLeft2()
            }else if x < 0 {
                moveRight2()
            }
        }
        
        
        
    }
}
