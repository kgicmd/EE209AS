//
//  ball.swift
//  pong
//
//  Created by Jojo Chen on 10/31/18.
//  Copyright © 2018 Jojo Chen. All rights reserved.
//

import SpriteKit

class Ball: SKNode{
    var circle = SKShapeNode()
    let radius: CGFloat = 15
    
    override init() {
        super.init()
        
        circle = SKShapeNode(circleOfRadius: radius)
        circle.fillColor = SKColor.red
        addChild(circle)
        
        physicsBody = SKPhysicsBody(circleOfRadius: radius)
        physicsBody?.isDynamic = true
        physicsBody?.affectedByGravity = false
        physicsBody?.allowsRotation = true
        physicsBody?.linearDamping = 0
        physicsBody?.angularDamping = 0
        physicsBody?.friction = 0
        physicsBody?.restitution = 1
        physicsBody?.categoryBitMask = BitMask.Ball
        physicsBody?.collisionBitMask = BitMask.ball
        physicsBody?.contactTestBitMask = BitMask.ball
        
        

        
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    
}
